import torch
from strategy import PropagationStrategy

class RSCStrategy(PropagationStrategy):
    def __init__(self, input_plane, output_plane, fft_strategy, wavelength):
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.wavelength = wavelength
        self.fft_strategy = fft_strategy
        self.transfer_function = self.get_transfer_function()

    def get_transfer_function(self):
        xx = self.fft_strategy.xx_input
        yy = self.fft_strategy.yy_input

        z = self.output_plane.center[-1] - self.input_plane.center[-1]
        shift = self.output_plane.center - self.input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]

        r = torch.sqrt((xx - x_shift)**2 + (yy - y_shift)**2 + z**2).real
        k = (2 * torch.pi / self.wavelength)

        h_rsc = torch.exp(torch.sign(z) * 1j * k * r) / r
        h_rsc *= ((1 / r) - (1j * k))
        h_rsc *= (1 / (2 * torch.pi))

        H = self.fft_strategy.fft2(h_rsc)

        mag = H.abs()
        ang = H.angle()
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j * ang)
        H = torch.reshape(H, (1, H.size(-2), H.size(-1)))
        H.requires_grad = False
        return H

    def propagate(self, input_wavefront):
        A = self.fft_strategy.fft2(input_wavefront)
        #A = torch.fft.fftshift(A)
        U = A * self.transfer_function
        U = self.fft_strategy.ifft2(U)
        #U = torch.fft.ifftshift(U, dim=(-1, -2))
        return U

