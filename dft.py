import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from loguru import logger
import pytorch_lightning as pl
#import cupy as cp

BACKENDS = {
    "numpy": np,
    "torch": torch,
    }


class DFT(pl.LightningModule):
    def __init__(self, input_plane, output_plane):
        super().__init__()
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.use_custom_dft = False
        self.build_dft_matrices(input_plane, output_plane)

    def build_dft_matrices(self, input_plane, output_plane):
        dx_input = input_plane.delta_x
        dy_input = input_plane.delta_y

        dx_output = output_plane.delta_x
        dy_output = output_plane.delta_y
        if dx_input.real == dx_output.real and dy_input.real == dy_output.real:
            logger.info("Using PyTorch DFT")
            self.use_custom_dft = False
            return
        logger.info("Using custom DFT")

        if dx_input.real < dx_output.real:
            fx = output_plane.fx_padded
        else:
            fx = input_plane.fx_padded

        if dy_input.real < dy_output.real:
            fy = output_plane.fy_padded
        else:
            fy = input_plane.fy_padded

        self.register_buffer("dft_matrix_x", torch.exp(-2j * torch.pi * torch.outer(fx, input_plane.x_padded)).unsqueeze(0))
        self.register_buffer("dft_matrix_y", torch.exp(-2j * torch.pi * torch.outer(fy, input_plane.y_padded)).unsqueeze(0))

    def forward(self, g):

        if not self.use_custom_dft:
            return torch.fft.fftn(g)
        else:
            # Make sure g is in B,M,N format
            if len(g.shape) == 4:
                b,c,w,h = g.shape
                g = g.squeeze()
                g = g.reshape(b,w,h)
            elif len(g.shape) != 3:
                g = g.squeeze()
                g = g.reshape(1,g.shape[-2],g.shape[-1])

            # Perform the DFT along x-axis
            g_dft_x = self.dft_matrix_x[0] @ g

            # Perform the DFT along y-axis
            g_dft_xy = self.dft_matrix_y[0] @ g_dft_x.permute(0, 2, 1)
            g_dft_xy = g_dft_xy.permute(0, 2, 1)
            return g_dft_xy


class DIFT(pl.LightningModule):
    def __init__(self, input_plane, output_plane):

        super().__init__()
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.use_custom_dift = False
        self.build_dift_matrices(input_plane, output_plane)

    def build_dift_matrices(self, input_plane, output_plane):

        # I might need to double these here
        M_output = output_plane.Nx
        N_output = output_plane.Ny

        M_input = input_plane.Nx
        N_input = input_plane.Ny

        dx_input = input_plane.delta_x
        dy_input = input_plane.delta_y

        dx_output = output_plane.delta_x
        dy_output = output_plane.delta_y

        if dx_input.real == dx_output.real and dy_input.real == dy_output.real and M_input == M_output and N_input == N_output:
            logger.info("Using PyTorch DIFT")
            self.use_custom_dift = False
            return
        logger.info("Using custom DIFT")

        if dx_input.real < dx_output.real:
            fx = output_plane.fx_padded
        else:
            fx = input_plane.fx_padded

        if dy_input.real < dy_output.real:
            fy = output_plane.fy_padded
        else:
            fy = input_plane.fy_padded

        self.register_buffer("dift_matrix_x", torch.exp(2j * torch.pi * torch.outer(output_plane.x_padded, fx)).unsqueeze(0) / M_output)
        self.register_buffer("dift_matrix_y", torch.exp(2j * torch.pi * torch.outer(output_plane.y_padded, fy)).unsqueeze(0) / N_output)

    def forward(self, x):
        if not self.use_custom_dift:
            return torch.fft.ifftn(x)
        else:
            # Make sure G is in B,M,N format
            if len(x.shape) == 4:
                b,c,w,h = x.shape
                x = x.squeeze()
                x = x.reshape(b,w,h)
            elif len(x.shape) != 3:
                x = x.squeeze()
                x = x.reshape(1,x.shape[-2],x.shape[-1])

            # Perform the DIFT using dot product along y-axis (columns)
            g_reconstructed_y = self.dift_matrix_y[0] @ x.permute(0, 2, 1)

            # Perform the DIFT using dot product along x-axis (rows)
            g_reconstructed = self.dift_matrix_x[0] @ g_reconstructed_y.permute(0, 2, 1)

            return g_reconstructed.unsqueeze(1)


def dft_1d(g, x, fx, dft_matrix = None, backend = BACKENDS["numpy"]):

    # Make sure g is in B,M format
    if len(g.shape) != 2:
        g = g.reshape(1,g.shape[-1])

    # Get the length of the input signal
    M = x.shape[0]

    # If the dft_matrix is not provided, create it.
    if dft_matrix is None:
        dft_matrix = backend.exp(-2j * backend.pi * backend.outer(fx, x))

        # Reshape the DFT matrix to properly broadcast during the dot product
        if backend == np:
            dft_matrix = dft_matrix.reshape(1, fx.shape[0], M)
        elif backend == torch:
            dft_matrix = dft_matrix.unsqueeze(0)  # Add batch dimension
            dft_matrix = dft_matrix.type(torch.complex128)

        # If the backend is torch, move it to the GPU if a device is available.
        if backend == torch and torch.cuda.is_available():
            dft_matrix = dft_matrix.to(g.device)

    # Perform the DFT using @ operator for batch processing
    if backend == np:
        result = dft_matrix[0] @ g.T
    elif backend == torch:
        result = dft_matrix[0] @ g.transpose(0, 1)

    return result.T

def dift_1d(G, x, fx, x_reconstruction, dift_matrix=None, backend=BACKENDS["numpy"]):
    # Make sure G is in B,M format
    if len(G.shape) != 2:
        G = G.reshape(1, G.shape[-1])

    # Get the length of the initial spatial signal
    M = x.shape[0]

    # If the dift_matrix is not provided, create it
    if dift_matrix is None:
        dift_matrix = backend.exp(2j * backend.pi * backend.outer(x_reconstruction, fx)) / M

        # Reshape the DIFT matrix to properly broadcast during the dot product
        if backend == np:
            dift_matrix = dift_matrix.reshape(1, x_reconstruction.shape[0], fx.shape[0])
            dift_matrix = dift_matrix.astype(complex)
        elif backend == torch:
            dift_matrix = dift_matrix.unsqueeze(0)  # Add batch dimension
            dift_matrix = dift_matrix.type(torch.complex128)

        # If the backend is torch, move it to the GPU if a device is available
        if backend == torch and torch.cuda.is_available():
            dift_matrix = dift_matrix.to(G.device)

    # Perform the DIFT using the @ operator for batch processing
    if backend == np:
        result = dift_matrix[0] @ G.T
    elif backend == torch:
        result = dift_matrix[0] @ G.transpose(0, 1)
    
    return result.T

def dft_2d(g, x, y, fx, fy, dft_matrix_x=None, dft_matrix_y=None, backend=BACKENDS["numpy"]):
        
    # Make sure g is in B,M,N format
    if len(g.shape) == 4:
        b,c,w,h = g.shape
        g = g.squeeze()
        g = g.reshape(b,w,h)
    elif len(g.shape) != 3:
        g = g.squeeze()
        g = g.reshape(1,g.shape[-2],g.shape[-1])

    # If the dft_matrix_x is not provided, create it
    if dft_matrix_x is None:
        M = x.shape[0]
        m = backend.arange(M)
        dft_matrix_x = backend.exp(-2j * backend.pi * backend.outer(fx, x))
        if backend == np:
            dft_matrix_x = dft_matrix_x.reshape(1, fx.shape[0], m.shape[0])
        elif backend == torch:
            dft_matrix_x = dft_matrix_x.unsqueeze(0)  # Add batch dimension

        # If the backend is torch, move it to the GPU if a device is available
        if backend == torch and torch.cuda.is_available():
            dft_matrix_x = dft_matrix_x.to(g.device)

    # If the dft_matrix_y is not provided, create it
    if dft_matrix_y is None:
        N = y.shape[0]
        n = backend.arange(N)
        dft_matrix_y = backend.exp(-2j * backend.pi * backend.outer(fy, y))
        if backend == np:
            dft_matrix_y = dft_matrix_y.reshape(1, fy.shape[0], n.shape[0])
        elif backend == torch:
            dft_matrix_y = dft_matrix_y.unsqueeze(0)  # Add batch dimension

        # If the backend is torch, move it to the GPU if a device is available
        if backend == torch and torch.cuda.is_available():
            dft_matrix_y = dft_matrix_y.to(g.device)

    # Perform the DFT along x-axis
    if backend == np:
        g_dft_x = dft_matrix_x[0] @ g.transpose(0, 2, 1)
    elif backend == torch:
        g_dft_x = dft_matrix_x[0] @ g

    # Perform the DFT along y-axis
    if backend == np:
        g_dft_xy = dft_matrix_y[0] @ g_dft_x.transpose(0, 2, 1)
        g_dft_xy = g_dft_xy.transpose(0, 2, 1)
    elif backend == torch:
        g_dft_xy = dft_matrix_y[0] @ g_dft_x.permute(0, 2, 1)
        g_dft_xy = g_dft_xy.permute(0, 2, 1)

    return g_dft_xy

def dift_2d(G, x, y, fx, fy, x_reconstruction, y_reconstruction, dift_matrix_x=None, dift_matrix_y=None, backend=BACKENDS["numpy"]):

    # Make sure G is in B,M,N format
    if len(G.shape) == 4:
        b,c,w,h = G.shape
        G = G.squeeze()
        G = G.reshape(b,w,h)
    elif len(G.shape) != 3:
        G = G.squeeze()
        G = G.reshape(1,G.shape[-2],G.shape[-1])

    # If the dift_matrix_x is not provided, create it
    if dift_matrix_x is None:
        M = x.shape[0]
        dift_matrix_x = backend.exp(2j * backend.pi * backend.outer(x_reconstruction, fx)) / M
        if backend == np:
            dift_matrix_x = dift_matrix_x.reshape(1, x_reconstruction.shape[0], fx.shape[0])
        elif backend == torch:
            dift_matrix_x = dift_matrix_x.unsqueeze(0)  # Add batch dimension

        # If the backend is torch, move it to the GPU if a device is available
        if backend == torch and torch.cuda.is_available():
            dift_matrix_x = dift_matrix_x.to(G.device)

    # If the dift_matrix_y is not provided, create it
    if dift_matrix_y is None:
        N = y.shape[0]
        dift_matrix_y = backend.exp(2j * backend.pi * backend.outer(y_reconstruction, fy)) / N
        if backend == np:
            dift_matrix_y = dift_matrix_y.reshape(1, y_reconstruction.shape[0], fy.shape[0])
        elif backend == torch:
            dift_matrix_y = dift_matrix_y.unsqueeze(0)  # Add batch dimension

        # If the backend is torch, move it to the GPU if a device is available
        if backend == torch and torch.cuda.is_available():
            dift_matrix_y = dift_matrix_y.to(G.device)

    # Perform the DIFT using dot product along y-axis (columns)
    if backend == np:
        g_reconstructed_y = dift_matrix_y[0] @ G.transpose(0, 2, 1)
    elif backend == torch:
        g_reconstructed_y = dift_matrix_y[0] @ G.permute(0, 2, 1)

    # Perform the DIFT using dot product along x-axis (rows)
    if backend == np:
        g_reconstructed = dift_matrix_x[0] @ g_reconstructed_y.transpose(0, 2, 1)
    elif backend == torch:
        g_reconstructed = dift_matrix_x[0] @ g_reconstructed_y.permute(0, 2, 1)

    return g_reconstructed.unsqueeze(1)


if __name__ == "__main__":

    backend_str = "torch"
    backend = BACKENDS[backend_str]

    # Define the spatial signal
    M = 1000
    x = backend.linspace(-1, 1, M, dtype=backend.complex128)

    # Create a signal to test the DFT
    # First, we will make a signal with 3 known frequencies
    g = backend.sin(2 * backend.pi * 1 * x) + 3*backend.sin(2 * backend.pi * 3 * x) + 5*backend.sin(2 * backend.pi * 5 * x)
    g = g.reshape(1,1,M)

    # Get the frequency domain - for now the whole range
    fx = np.fft.fftfreq(len(x), np.diff(x)[0])
    fx = np.fft.fftshift(fx)

    if backend == torch:
        fx = torch.tensor(fx)

    # Perform the DFT with numpy and the chosen backend
    G_backend = dft_1d(g, x, fx, backend = backend)

    G_numpy = np.fft.fft(g.squeeze())
    G_numpy = np.fft.fftshift(G_numpy)

    # Define the normalization factor
    norm_factor = M/2 

    # Get the normalized amplitudes
    G_backend_amp = backend.abs(G_backend).squeeze()/norm_factor
    G_numpy_amp = np.abs(G_numpy).squeeze()/norm_factor

    # Get the difference between the two
    diff = G_backend_amp - G_numpy_amp

    # Plot them to compare
    fig, ax = plt.subplots(3,1, figsize = (8, 7.5))
    ax[0].plot(x, g.squeeze().real, label = "Original signal")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("g(x)")
    ax[0].set_title("Original signal")

    ax[1].plot(fx, G_backend_amp, label = "Custom DFT with backend = {}".format(backend_str))
    ax[1].plot(fx, G_numpy_amp, label = "DFT with numpy")
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Magnitude")
    ax[1].set_title("Magnitude of the DFT")

    ax[2].plot(fx, diff, label = "Difference")
    ax[2].set_xlabel("Frequency")
    ax[2].set_ylabel("Magnitude")
    ax[2].set_title("Difference between the two")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.tight_layout()
    plt.show()

    fig.savefig("dft_1d.pdf")

    # Now we will test the DIFT
    # Define the reconstruction window - for now the same size and discretization as the original signal
    x_reconstruction = backend.linspace(-1, 1, M, dtype=backend.complex128)

    # Shift the numpy signal
    G_numpy = np.fft.ifftshift(G_numpy)

    # Perform the DIFT with numpy and the chosen backend
    g_reconstructed_backend = dift_1d(G_backend, x, fx, x_reconstruction, backend = backend).squeeze()
    g_reconstructed_numpy = np.fft.ifft(G_numpy).squeeze()

    # Shift them

    # Get the difference between the two
    diff = g_reconstructed_backend - g_reconstructed_numpy

    # Plot them to compare
    fig, ax = plt.subplots(2,1, figsize = (8, 7.5))

    ax[0].plot(x, g.squeeze().real, label = "Original signal")
    ax[0].plot(x_reconstruction, g_reconstructed_backend.real, label = "Custom DIFT with backend = {}".format(backend_str))
    ax[0].plot(x_reconstruction, g_reconstructed_numpy.real, label = "DIFT with numpy")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("g(x)")
    ax[0].set_title("Reconstructed signal")

    ax[1].plot(x_reconstruction, diff.real, label = "Difference")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Magnitude")
    ax[1].set_title("Difference between the two")
    
    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    fig.savefig("dift_1d.pdf")

    # Now, let's look at a custom frequency range
    fx_smaller = np.arange(-15, 15, 1, dtype=np.complex128)
    if backend == torch:
        fx_smaller = torch.tensor(fx_smaller)

    # Perform the DFT with numpy and the chosen backend
    G_backend = dft_1d(g, x, fx_smaller, backend = backend)

    # Perform the DFT with numpy
    G_numpy = np.fft.fft(g.squeeze())
    G_numpy = np.fft.fftshift(G_numpy)

    # Get the normalized amplitudes
    G_backend_amp = backend.abs(G_backend).squeeze()/norm_factor
    G_numpy_amp = np.abs(G_numpy).squeeze()/norm_factor

    # Plot them to compare
    fig, ax = plt.subplots(2,1, figsize = (8, 5))
    ax[0].plot(x, g.squeeze().real, label = "Original signal")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("g(x)")
    ax[0].set_title("Original signal")

    ax[1].plot(fx_smaller, G_backend_amp, label = "Custom DFT with backend = {}".format(backend_str))
    ax[1].plot(fx, G_numpy_amp, label = "DFT with numpy")
    ax[1].set_xlim(-15, 15)
    ax[1].set_xlabel("Frequency")
    ax[1].set_ylabel("Magnitude")
    ax[1].set_title("Magnitude of the DFT")

    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    fig.savefig("dft_1d_smaller.pdf")

    # Now reconstruct the signal in a smaller window and smaller discretization
    x_reconstruction = backend.linspace(-0.5, 0.5, M, dtype=backend.complex128)

    # Perform the DIFT with numpy and the chosen backend
    g_reconstructed_backend = dift_1d(G_backend, x, fx_smaller, x_reconstruction, backend = backend)

    # Perform the DIFT with numpy
    G_numpy = np.fft.ifftshift(G_numpy)
    g_reconstructed_numpy = np.fft.ifft(G_numpy)

    # Plot them to compare
    fig, ax = plt.subplots(1,1, figsize = (8, 5))

    ax.plot(x, g.squeeze().real, label = "Original signal")
    ax.plot(x, g_reconstructed_numpy.squeeze().real,linestyle='dashed', label = "DIFT with numpy")
    ax.plot(x_reconstruction, g_reconstructed_backend.squeeze().real, label = "Custom DIFT with backend = {}".format(backend_str))
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.set_title("Reconstructed signal")

    ax.legend()
    
    plt.tight_layout()
    plt.show()

    fig.savefig("dift_1d_smaller.pdf")


    # Now we will test the 2D DFT
    # Define the spatial signal
    M = 100
    N = 100
    x = np.linspace(-1, 1, M, dtype=np.complex128)
    y = np.linspace(-1, 1, N, dtype=np.complex128)
    X, Y = np.meshgrid(x, y)

    # Get the frequency domain - for now the whole range
    fx = np.fft.fftfreq(len(x), np.diff(x)[0])
    fy = np.fft.fftfreq(len(y), np.diff(y)[0])
    fx = np.fft.fftshift(fx)
    fy = np.fft.fftshift(fy)
    fxx, fyy = np.meshgrid(fx, fy)
    
    if backend == torch:
        x = torch.tensor(x)
        y = torch.tensor(y)
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        fx = torch.tensor(fx)
        fy = torch.tensor(fy)
        fxx = torch.tensor(fxx)
        fyy = torch.tensor(fyy)

    # Create a signal to test the DFT
    # First, we will make a signal with 3 known frequencies
    g = backend.sin(2 * backend.pi * 1 * X) + 3*backend.sin(2 * backend.pi * 3 * X) + 5*backend.sin(2 * backend.pi * 5 * X)
    g += backend.sin(2 * backend.pi * 1 * Y) + 3*backend.sin(2 * backend.pi * 3 * Y) + 5*backend.sin(2 * backend.pi * 5 * Y)
    g = g.reshape(1,M,N)

    # Perform the DFT with numpy and the chosen backend
    G_backend = dft_2d(g, x, y, fx, fy, backend = backend)

    G_numpy = np.fft.fft2(g.squeeze())
    G_numpy = np.fft.fftshift(G_numpy)

    # Define the normalization factor
    norm_factor = M*N/2

    # Get the normalized amplitudes
    G_backend_amp = backend.abs(G_backend.squeeze())/norm_factor
    G_numpy_amp = np.abs(G_numpy.squeeze())/norm_factor

    # Get the difference between the two
    diff = G_backend_amp - G_numpy_amp

    # Plot them to compare
    fig, ax = plt.subplots(1,4, figsize = (16,4))
    im0 = ax[0].imshow(g.squeeze().numpy().real, extent = [x[0].real, x[-1].real, y[0].real, y[-1].real])
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Original signal")
    # Create the colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax=cax)

    im1 = ax[1].imshow(G_backend_amp, extent = [fx[0].real, fx[-1].real, fy[0].real, fy[-1].real])
    ax[1].set_xlabel("Frequency x")
    ax[1].set_ylabel("Frequency y")
    ax[1].set_title("Custom 2D DFT \nbackend = {}".format(backend_str))
    # Create the colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)

    im2 = ax[2].imshow(G_numpy_amp, extent = [fx[0].real, fx[-1].real, fy[0].real, fy[-1].real])
    ax[2].set_xlabel("Frequency x")
    ax[2].set_ylabel("Frequency y")
    ax[2].set_title("Magnitude of the 2D DFT \nnumpy")
    # Create the colorbar
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax)

    im3 = ax[3].imshow(diff, extent = [fx[0].real, fx[-1].real, fy[0].real, fy[-1].real])
    ax[3].set_xlabel("Frequency x")
    ax[3].set_ylabel("Frequency y")
    ax[3].set_title("Difference")
    # Create the colorbar
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax)

    # Set the aspects to equal
    for a in ax:
        a.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    fig.savefig("dft_2d.pdf")


    # Now we will test the 2D DIFT

    # Define the reconstruction window - for now the same size and discretization as the original signal
    x_reconstruction = backend.linspace(-1, 1, M, dtype=backend.complex128)
    y_reconstruction = backend.linspace(-1, 1, N, dtype=backend.complex128)

    # Perform the DIFT with numpy and the chosen backend
    g_reconstructed_backend = dift_2d(G_backend, x, y, fx, fy, x_reconstruction, y_reconstruction, backend = backend)
    g_reconstructed_numpy = np.fft.ifft2(np.fft.ifftshift(G_numpy.squeeze()))

    # Get the difference between the two
    diff = g_reconstructed_backend - g_reconstructed_numpy

    # Plot them to compare
    fig, ax = plt.subplots(1,4, figsize = (16,4))
    im0 = ax[0].imshow(g.squeeze().numpy().real, extent = [x[0].real, x[-1].real, y[0].real, y[-1].real]) 
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Original signal")
    # Create the colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax=cax)

    im1 = ax[1].imshow(g_reconstructed_backend.squeeze().real, extent = [x_reconstruction[0].real, x_reconstruction[-1].real, y_reconstruction[0].real, y_reconstruction[-1].real])
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Custom 2D DIFT \nbackend = {}".format(backend_str))
    # Create the colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)

    im2 = ax[2].imshow(g_reconstructed_numpy.squeeze().real, extent = [x_reconstruction[0].real, x_reconstruction[-1].real, y_reconstruction[0].real, y_reconstruction[-1].real]) 
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[2].set_title("DIFT with numpy")
    # Create the colorbar
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax)

    im3 = ax[3].imshow(diff.squeeze().real, extent = [x_reconstruction[0].real, x_reconstruction[-1].real, y_reconstruction[0].real, y_reconstruction[-1].real]) 
    ax[3].set_xlabel("x")
    ax[3].set_ylabel("y")
    ax[3].set_title("Difference")
    # Create the colorbar
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax)

    # Set the aspects to equal
    for a in ax:
        a.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    fig.savefig("dift_2d.pdf")

   
    # Let's look at a test image
    # Load the image
    img = Image.open("data/STI/Special/Lenstarg.png").convert("L")

    # Convert the image to a numpy/cupy array or a torch tensor
    if backend == torch:
        img = torch.tensor(np.array(img)).type(torch.complex128)
    else:
        img = backend.array(img).astype(np.complex128)

    # This test image is 2168x7086, where the 7086 corresponds to 2.5mm
    # This makes the pixel size 2.5/7086 mm, and the first dimension = 0.765mm
    # It is also [0-255] grayscale, so we need to normalize it to [0, 1]
    # We also need to swap the axes to have the spatial dimensions in the correct order

    # Normalize the image to [0, 1]
    img = img/255

    # Swap the axes
    img = img.T

    # Make the shape of the image B,M,N
    img = img.reshape(1, img.shape[-2], img.shape[-1])

    # Create the spatial dimensions
    x = np.linspace(-2.5/2, 2.5/2, img.shape[-2], dtype=np.complex128)
    y = np.linspace(-0.765/2, 0.765/2, img.shape[-1], dtype=np.complex128)
    X, Y = np.meshgrid(x, y)

    # Get the frequency domain - for now the whole range
    fx = np.fft.fftfreq(len(x), np.diff(x)[0])
    fx = np.fft.fftshift(fx)
    fy = np.fft.fftfreq(len(y), np.diff(y)[0])
    fy = np.fft.fftshift(fy)
    fxx, fyy = np.meshgrid(fx, fy)

    if backend == torch:
        x = torch.tensor(x)
        y = torch.tensor(y)
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        fx = torch.tensor(fx)
        fy = torch.tensor(fy)
        fxx = torch.tensor(fxx)
        fyy = torch.tensor(fyy)

    # Perform the DFT with numpy and the chosen backend
    G_backend = dft_2d(img, x, y, fx, fy, backend = backend)

    G_numpy = np.fft.fft2(img.squeeze())
    G_numpy = np.fft.fftshift(G_numpy)

    # Define the normalization factor
    norm_factor = img.shape[-2]*img.shape[-1]/2
    #norm_factor = 1

    # Get the normalized amplitudes
    G_backend_amp = backend.abs(G_backend.squeeze())/norm_factor
    G_numpy_amp = np.abs(G_numpy.squeeze())/norm_factor

    # Get the difference between the two
    diff = G_backend_amp - G_numpy_amp

    # Plot them to compare
    fig, ax = plt.subplots(4,1, figsize = (10, 10))
    im0 = ax[0].imshow(img.squeeze().real.T, extent = [x[0].real, x[-1].real, y[0].real, y[-1].real]) 
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Original image")
    # Create the colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax=cax)

    im1 = ax[1].imshow(G_backend_amp, extent = [fx[0].real, fx[-1].real, fy[0].real, fy[-1].real]) 
    ax[1].set_xlabel("Frequency x")
    ax[1].set_ylabel("Frequency y")
    ax[1].set_title("Custom 2D DFT \nbackend = {}".format(backend_str))
    # Create the colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)

    im2 = ax[2].imshow(G_numpy_amp, extent = [fx[0].real, fx[-1].real, fy[0].real, fy[-1].real]) 
    ax[2].set_xlabel("Frequency x")
    ax[2].set_ylabel("Frequency y")
    ax[2].set_title("Magnitude of the 2D DFT \nnumpy")
    # Create the colorbar
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax)

    im3 = ax[3].imshow(diff, extent = [fx[0].real, fx[-1].real, fy[0].real, fy[-1].real]) 
    ax[3].set_xlabel("Frequency x")
    ax[3].set_ylabel("Frequency y")
    ax[3].set_title("Difference")
    # Create the colorbar
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax)

    # Set the aspects to equal
    for a in ax:
        a.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    
    fig.savefig("dft_2d_image.pdf")


    # Now we will test the 2D DIFT

    # Define the reconstruction window - for now the same size and discretization as the original signal
    x_reconstruction = backend.linspace(-2.5/2, 2.5/2, img.shape[-2], dtype=backend.complex128)
    y_reconstruction = backend.linspace(-0.765/2, 0.765/2, img.shape[-1], dtype=backend.complex128)

    # Perform the DIFT with numpy and the chosen backend
    g_reconstructed_backend = dift_2d(G_backend, x, y, fx, fy, x_reconstruction, y_reconstruction, backend = backend)
    g_reconstructed_numpy = np.fft.ifft2(np.fft.ifftshift(G_numpy))

    # Get the difference between the two
    diff = g_reconstructed_backend.squeeze() - g_reconstructed_numpy.squeeze()

    # Plot them to compare
    fig, ax = plt.subplots(4,1, figsize = (10, 10))
    im0 = ax[0].imshow(img.squeeze().real.T, extent = [x[0].real, x[-1].real, y[0].real, y[-1].real])
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Original image")
    # Create the colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax=cax)

    im1 = ax[1].imshow(g_reconstructed_backend.squeeze().real.T, extent = [x_reconstruction[0].real, x_reconstruction[-1].real, y_reconstruction[0].real, y_reconstruction[-1].real]) 
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Custom 2D DIFT \nbackend = {}".format(backend_str))
    # Create the colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)

    im2 = ax[2].imshow(g_reconstructed_numpy.squeeze().real.T, extent = [x_reconstruction[0].real, x_reconstruction[-1].real, y_reconstruction[0].real, y_reconstruction[-1].real]) 
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    ax[2].set_title("DIFT with numpy")
    # Create the colorbar
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax)

    im3 = ax[3].imshow(diff.squeeze().real.T, extent = [x_reconstruction[0].real, x_reconstruction[-1].real, y_reconstruction[0].real, y_reconstruction[-1].real]) 
    ax[3].set_xlabel("x")
    ax[3].set_ylabel("y")
    ax[3].set_title("Difference")
    # Create the colorbar
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax)

    plt.tight_layout()
    plt.show()

    fig.savefig("dift_2d_image.pdf")

    # Let's reconstruct to a smaller window with smaller discretization
    x_reconstruction_smaller = backend.linspace(1.21, 1.23, 2000, dtype=backend.complex128)
    y_reconstruction_smaller = backend.linspace(-0.765/2, 0.765/2, 2168, dtype=backend.complex128)

    # Perform the DIFT with our custom function and chosen backend
    g_reconstructed_backend = dift_2d(G_backend, x, y, fx, fy, x_reconstruction_smaller, y_reconstruction_smaller, backend = backend).squeeze()

    # Perform the DIFT with numpy   
    g_reconstructed_numpy = np.fft.ifft2(np.fft.ifftshift(G_numpy.squeeze()))

    # Get the number of points in the larger window when cropped to the smaller window
    num_points = len(np.where((x_reconstruction.real >= x_reconstruction_smaller[0].real) & (x_reconstruction.real <= x_reconstruction_smaller[-1].real))[0])

    # Plot them to compare - we will manually limit the numpy reconstruction to the same window

    fig, ax = plt.subplots(1,1, figsize = (8, 5))

    ln0 = ax.plot(x_reconstruction_smaller.real, g_reconstructed_backend[:,2168//2].real, label = "Custom DIFT with backend = {}".format(backend_str))
    ln1 = ax.plot(x_reconstruction.real, g_reconstructed_numpy[:,2168//2].real, linestyle='dashed', label = "DIFT with numpy")
    ax.scatter(x_reconstruction.real, g_reconstructed_numpy[:,2168//2].real, color = "red", s = 10, label='Numpy samples (N={})'.format(num_points))
    ax.set_xlim(x_reconstruction_smaller[0].real, x_reconstruction_smaller[-1].real)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    fig.savefig("dift_2d_image_smaller.pdf")




