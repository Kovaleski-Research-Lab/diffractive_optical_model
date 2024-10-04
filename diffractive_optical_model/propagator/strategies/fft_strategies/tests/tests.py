import matplotlib.pyplot as plt
import unittest
import os
import sys
import torch
import numpy as np
import time
import loguru
import yaml

sys.path.append('../') # To get the strategies
sys.path.append('../../../../') # To get the plane


# Create some plane paramters for the tests

params_plane0 = {
    'name': 'plane0',
    'center': [0,0,0],
    'size': [8.64e+3, 8.64e+3],
    'normal': [0,0,1],
    'Nx': 1080,
    'Ny': 1080,
    }

params_plane1 = {
    'name': 'plane1',
    'center': [0,0,0],
    'size': [8.64e+3, 8.64e+3],
    'normal': [0,0,1],
    'Nx': 1080,
    'Ny': 540,
    }

params_plane2 = {
    'name': 'plane2',
    'center': [0,0,0],
    'size': [8.64e+3, 8.64e+3],
    'normal': [0,0,1],
    'Nx': 540,
    'Ny': 1080,
    }

params_plane3 = {
        'name': 'plane3',
        'center': [0,0,0],
        'size': [8.64e+3, 8.64e+3],
        'normal': [0,0,1],
        'Nx': 540,
        'Ny': 540,
        }

params_plane4 = {
        'name': 'plane4',
        'center': [0,0,0],
        'size': [3e+3, 8.64e+3],
        'normal': [0,0,1],
        'Nx': 1080,
        'Ny': 1080,
        }

params_plane5 = {
        'name': 'plane5',
        'center': [0,0,0],
        'size': [8.64e+3, 3e+3],
        'normal': [0,0,1],
        'Nx': 1080,
        'Ny': 1080,
        }

params_plane6 = {
        'name': 'plane6',
        'center': [0,0,0],
        'size': [3e+3, 3e+3],
        'normal': [0,0,1],
        'Nx': 1080,
        'Ny': 1080,
        }

params_plane7 = {
        'name': 'plane7',
        'center': [0,0,0],
        'size': [8, 8],
        'normal': [0,0,1],
        'Nx': 1080,
        'Ny': 1080,
        }

from plane.plane import Plane
from mp_strategy import MPFFTStrategy
from pytorch_strategy import PyTorchFFTStrategy
from numpy_strategy import NumpyFFTStrategy

class TestDFT(unittest.TestCase):
    def setup(self):
        pass
    def tearDown(self):
        pass

    def test_init_mpfft(self):
        # Create a plane
        plane = Plane(params_plane0)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane, plane)

    def test_init_mpfft_different_ny(self):
        # Create a plane
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

    def test_init_mpfft_different_nx(self):
        # Create a plane
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane2)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

    def test_init_mpfft_different_nx_ny(self):
        # Create a plane
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane3)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

    def test_init_mpfft_different_x_size(self):
        # Create a plane
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane4)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

    def test_init_mpfft_different_y_size(self):
        # Create a plane
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane5)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

    def test_init_mpfft_different_x_y_size(self):
        # Create a plane
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane6)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

    def test_fx_fy_picking_nopad(self):
        kwargs = {'padded': False}
        # Create a plane
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)
        plane2 = Plane(params_plane2)
        plane3 = Plane(params_plane3)
        plane4 = Plane(params_plane4)
        plane5 = Plane(params_plane5)
        plane6 = Plane(params_plane6)

        # Initialize the strategies
        strategy0 = MPFFTStrategy(plane0, plane1, kwargs)
        strategy1 = MPFFTStrategy(plane0, plane2, kwargs)
        strategy2 = MPFFTStrategy(plane0, plane3, kwargs)
        strategy3 = MPFFTStrategy(plane0, plane4, kwargs)
        strategy4 = MPFFTStrategy(plane0, plane5, kwargs)
        strategy5 = MPFFTStrategy(plane0, plane6, kwargs)

        # The fx and fy should be picked from the plane with the largest delta_x and delta_y
        if plane0.delta_x.real <= plane1.delta_x.real:
            self.assertTrue(np.allclose(strategy0.fx, plane1.fx))
        else:
            self.assertTrue(np.allclose(strategy0.fx, plane0.fx))

        if plane0.delta_y.real <= plane1.delta_y.real:
            self.assertTrue(np.allclose(strategy0.fy, plane1.fy))
        else:
            self.assertTrue(np.allclose(strategy0.fy, plane0.fy))

        if plane0.delta_x.real <= plane2.delta_x.real:
            self.assertTrue(np.allclose(strategy1.fx, plane2.fx))
        else:
            self.assertTrue(np.allclose(strategy1.fx, plane0.fx))

        if plane0.delta_y.real <= plane2.delta_y.real:
            self.assertTrue(np.allclose(strategy1.fy, plane2.fy))
        else:
            self.assertTrue(np.allclose(strategy1.fy, plane0.fy))

        if plane0.delta_x.real <= plane3.delta_x.real:
            self.assertTrue(np.allclose(strategy2.fx, plane3.fx))
        else:
            self.assertTrue(np.allclose(strategy2.fx, plane0.fx))

        if plane0.delta_y.real <= plane3.delta_y.real:
            self.assertTrue(np.allclose(strategy2.fy, plane3.fy))
        else:
            self.assertTrue(np.allclose(strategy2.fy, plane0.fy))

        if plane0.delta_x.real <= plane4.delta_x.real:
            self.assertTrue(np.allclose(strategy3.fx, plane4.fx))
        else:
            self.assertTrue(np.allclose(strategy3.fx, plane0.fx))

        if plane0.delta_y.real <= plane4.delta_y.real:
            self.assertTrue(np.allclose(strategy3.fy, plane4.fy))
        else:
            self.assertTrue(np.allclose(strategy3.fy, plane0.fy))

        if plane0.delta_x.real <= plane5.delta_x.real:
            self.assertTrue(np.allclose(strategy4.fx, plane5.fx))
        else:
            self.assertTrue(np.allclose(strategy4.fx, plane0.fx))

        if plane0.delta_y.real <= plane5.delta_y.real:
            self.assertTrue(np.allclose(strategy4.fy, plane5.fy))
        else:
            self.assertTrue(np.allclose(strategy4.fy, plane0.fy))

        if plane0.delta_x.real <= plane6.delta_x.real:
            self.assertTrue(np.allclose(strategy5.fx, plane6.fx))
        else:
            self.assertTrue(np.allclose(strategy5.fx, plane0.fx))

        if plane0.delta_y.real <= plane6.delta_y.real:
            self.assertTrue(np.allclose(strategy5.fy, plane6.fy))
        else:
            self.assertTrue(np.allclose(strategy5.fy, plane0.fy))

    def test_fx_fy_picking_pad(self):
        # Create the planes
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)
        plane2 = Plane(params_plane2)
        plane3 = Plane(params_plane3)
        plane4 = Plane(params_plane4)
        plane5 = Plane(params_plane5)
        plane6 = Plane(params_plane6)

        kwargs = {'padded': True}

        # Initialize the strategies
        strategy0 = MPFFTStrategy(plane0, plane1, kwargs)
        strategy1 = MPFFTStrategy(plane0, plane2, kwargs)
        strategy2 = MPFFTStrategy(plane0, plane3, kwargs)
        strategy3 = MPFFTStrategy(plane0, plane4, kwargs)
        strategy4 = MPFFTStrategy(plane0, plane5, kwargs)
        strategy5 = MPFFTStrategy(plane0, plane6, kwargs)

        # The fx and fy should be picked from the plane with the largest delta_x and delta_y
        if plane0.delta_x.real <= plane1.delta_x.real:
            self.assertTrue(np.allclose(strategy0.fx, plane1.fx_padded))
        else:
            self.assertTrue(np.allclose(strategy0.fx, plane0.fx_padded))

        if plane0.delta_y.real <= plane1.delta_y.real:
            self.assertTrue(np.allclose(strategy0.fy, plane1.fy_padded))
        else:
            self.assertTrue(np.allclose(strategy0.fy, plane0.fy_padded))

        if plane0.delta_x.real <= plane2.delta_x.real:
            self.assertTrue(np.allclose(strategy1.fx, plane2.fx_padded))
        else:
            self.assertTrue(np.allclose(strategy1.fx, plane0.fx_padded))

        if plane0.delta_y.real <= plane2.delta_y.real:
            self.assertTrue(np.allclose(strategy1.fy, plane2.fy_padded))
        else:
            self.assertTrue(np.allclose(strategy1.fy, plane0.fy_padded))

        if plane0.delta_x.real <= plane3.delta_x.real:
            self.assertTrue(np.allclose(strategy2.fx, plane3.fx_padded))
        else:
            self.assertTrue(np.allclose(strategy2.fx, plane0.fx_padded))

        if plane0.delta_y.real <= plane3.delta_y.real:
            self.assertTrue(np.allclose(strategy2.fy, plane3.fy_padded))
        else:
            self.assertTrue(np.allclose(strategy2.fy, plane0.fy_padded))

        if plane0.delta_x.real <= plane4.delta_x.real:
            self.assertTrue(np.allclose(strategy3.fx, plane4.fx_padded))
        else:
            self.assertTrue(np.allclose(strategy3.fx, plane0.fx_padded))

        if plane0.delta_y.real <= plane4.delta_y.real:
            self.assertTrue(np.allclose(strategy3.fy, plane4.fy_padded))
        else:
            self.assertTrue(np.allclose(strategy3.fy, plane0.fy_padded))

        if plane0.delta_x.real <= plane5.delta_x.real:
            self.assertTrue(np.allclose(strategy4.fx, plane5.fx_padded))
        else:
            self.assertTrue(np.allclose(strategy4.fx, plane0.fx_padded))

        if plane0.delta_y.real <= plane5.delta_y.real:
            self.assertTrue(np.allclose(strategy4.fy, plane5.fy_padded))
        else:
            self.assertTrue(np.allclose(strategy4.fy, plane0.fy_padded))

        if plane0.delta_x.real <= plane6.delta_x.real:
            self.assertTrue(np.allclose(strategy5.fx, plane6.fx_padded))
        else:
            self.assertTrue(np.allclose(strategy5.fx, plane0.fx_padded))

        if plane0.delta_y.real <= plane6.delta_y.real:
            self.assertTrue(np.allclose(strategy5.fy, plane6.fy_padded))
        else:
            self.assertTrue(np.allclose(strategy5.fy, plane0.fy_padded))


    def test_dft_matrix_dtype(self):
        # Create planes with different precision
        plane0 = Plane(params_plane0, bits=64)
        plane1 = Plane(params_plane1, bits=64)
        plane2 = Plane(params_plane0, bits=128)
        plane3 = Plane(params_plane1, bits=128)

        # Initialize the strategy
        strategy0 = MPFFTStrategy(plane0, plane1)
        strategy1 = MPFFTStrategy(plane2, plane3)

        self.assertTrue(strategy0.dft_matrix_x.dtype == torch.complex64)
        self.assertTrue(strategy0.dft_matrix_y.dtype == torch.complex64)
        self.assertTrue(strategy0.idft_matrix_x.dtype == torch.complex64)
        self.assertTrue(strategy0.idft_matrix_y.dtype == torch.complex64)

        self.assertTrue(strategy1.dft_matrix_x.dtype == torch.complex128)
        self.assertTrue(strategy1.dft_matrix_y.dtype == torch.complex128)
        self.assertTrue(strategy1.idft_matrix_x.dtype == torch.complex128)
        self.assertTrue(strategy1.idft_matrix_y.dtype == torch.complex128)

    def test_np_torch_dft(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)

        # Initialize the numpy and pytorch strategies
        strategy_np = NumpyFFTStrategy()
        strategy_torch = PyTorchFFTStrategy(plane0, plane0)

        # Create a signal with known frequencies in 1D
        x = plane0.x
        g = np.sin(2*np.pi*x / 1.e+3) + np.sin(4*np.pi*x / 1.e+3) + np.sin(6*np.pi*x / 1.e+3)

        # Perform the DFT using numpy
        g_dft_np = strategy_np.fft(g)

        # Perform the DFT using torch
        g_dft_torch = strategy_torch.fft(g)

        magnitude_np = np.abs(g_dft_np)
        magnitude_torch = torch.abs(g_dft_torch)

        # Check if the results are the same
        self.assertTrue(np.allclose(magnitude_np, magnitude_torch.numpy()))

    def test_dft_matrix_precisions_lowtol(self):
        # Make a plane with 64 bits
        plane0 = Plane(params_plane0, bits=64)

        # Make a plane with 128 bits
        plane1 = Plane(params_plane0, bits=128)

        # Initialize two MPFFT strategies
        strategy0 = MPFFTStrategy(plane0, plane0)
        strategy1 = MPFFTStrategy(plane1, plane1)

        # Check if the DFT matrices between the two are the same
        self.assertTrue(np.allclose(strategy0.dft_matrix_x.numpy(), strategy1.dft_matrix_x.numpy(), atol=1e-1))
        self.assertTrue(np.allclose(strategy0.dft_matrix_y.numpy(), strategy1.dft_matrix_y.numpy(), atol=1e-1))

    def test_torch_mpfft_1d_nice_nopad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane0)

        # Create an input signal with known frequencies in 1D
        x = plane0.x
        g = torch.sin(2*torch.pi*x / 1.e+3) + torch.sin(4*torch.pi*x / 1.e+3) + torch.sin(6*torch.pi*x / 1.e+3)
        g = g.unsqueeze(0).to(torch.complex128)

        # Perform the DFT
        g_dft_mp = strategy.fft(g)
        g_dft_mp = torch.fft.fftshift(g_dft_mp, dim=-1)

        # Perform the DFT using torch
        g_dft_torch = torch.fft.fft(g)
        g_dft_torch = torch.fft.fftshift(g_dft_torch, dim=-1)

        # Get the magnitude
        magnitude_torch = torch.abs(g_dft_torch)
        magnitude_mp = torch.abs(g_dft_mp)

        # Check if the results are the same
        self.assertTrue(np.allclose(magnitude_torch.numpy(), magnitude_mp.numpy()))

    def test_torch_mpfft_1d_nice_pad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)

        kwargs = {'padded': True}
        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane0, kwargs)

        # Create an input signal with known frequencies in 1D
        x = plane0.x
        g = torch.sin(2*torch.pi*x / 1.e+3) + torch.sin(4*torch.pi*x / 1.e+3) + torch.sin(6*torch.pi*x / 1.e+3)
        g_padded = torch.zeros(1, plane0.Nx*2).to(torch.complex128)
        g_padded[0, plane0.Nx//2:plane0.Nx+plane0.Nx//2] = g.to(torch.complex128)

        # Perform the DFT
        g_dft_mp = strategy.fft(g_padded)
        g_dft_mp = torch.fft.fftshift(g_dft_mp, dim=-1)

        # Perform the DFT using torch
        g_dft_torch = torch.fft.fft(g_padded)
        g_dft_torch = torch.fft.fftshift(g_dft_torch, dim=-1)

        # Get the magnitude
        magnitude_torch = torch.abs(g_dft_torch)
        magnitude_mp = torch.abs(g_dft_mp)

        # Check if the results are the same
        self.assertTrue(np.allclose(magnitude_torch.numpy(), magnitude_mp.numpy()))


    def test_torch_mpfft_1d_random_nopad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane0)

        # Create a random input
        g = torch.randn(1, plane0.Nx).to(torch.complex128) 

        # Perform the DFT
        g_dft = strategy.fft(g)

        # Perform the DFT using torch
        g_dft_torch = torch.fft.fft(g)

        # Get the magnitude
        magnitude_torch = torch.abs(g_dft_torch)
        magnitude_mp = torch.abs(g_dft)

        # Check if the results are the same
        self.assertTrue(np.allclose(magnitude_torch.numpy(), magnitude_mp.numpy()))

    def test_torch_mpfft_1d_random_pad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)

        kwargs = {'padded': True}
        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane0, kwargs)

        # Create an input signal with known frequencies in 1D
        g_padded = torch.randn(1, plane0.Nx*2).to(torch.complex128)

        # Perform the DFT
        g_dft_mp = strategy.fft(g_padded)
        g_dft_mp = torch.fft.fftshift(g_dft_mp, dim=-1)

        # Perform the DFT using torch
        g_dft_torch = torch.fft.fft(g_padded)
        g_dft_torch = torch.fft.fftshift(g_dft_torch, dim=-1)

        # Get the magnitude
        magnitude_torch = torch.abs(g_dft_torch)
        magnitude_mp = torch.abs(g_dft_mp)

        # Check if the results are the same
        self.assertTrue(np.allclose(magnitude_torch.numpy(), magnitude_mp.numpy()))

    def test_torch_mpfft_2d_nice_nopad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)
        plane1 = Plane(params_plane0, bits=128)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

        # Create an input signal with known frequencies in 2D
        xx, yy = plane0.xx, plane0.yy
        g = torch.sin(2*torch.pi*xx / 1.e+3) + torch.sin(4*torch.pi*yy / 1.e+3) + torch.sin(6*torch.pi*xx / 1.e+3).to(torch.complex128)

        # Perform the DFT
        g_dft = strategy.fft2(g)

        # Perform the DFT using torch
        g_dft_torch = torch.fft.fftn(g)

        # Get the magnitude
        magnitude_torch = torch.abs(g_dft_torch)
        magnitude_mp = torch.abs(g_dft)

        # Check if the results are the same
        self.assertTrue(np.allclose(magnitude_torch.numpy(), magnitude_mp.numpy(), atol=1e-6))

    def test_torch_mpfft_2d_nice_pad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)
        plane1 = Plane(params_plane0, bits=128)

        kwargs = {'padded': True}
        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Create an input signal with known frequencies in 2D
        xx, yy = plane0.xx, plane0.yy
        g = torch.sin(2*torch.pi*xx / 1.e+3) + torch.sin(4*torch.pi*yy / 1.e+3) + torch.sin(6*torch.pi*xx / 1.e+3).to(torch.complex128)
        g_padded = torch.zeros(1, plane0.Nx*2, plane0.Ny*2).to(torch.complex128)
        g_padded[0, plane0.Nx//2:plane0.Nx+plane0.Nx//2, plane0.Ny//2:plane0.Ny+plane0.Ny//2] = g

        # Perform the DFT
        g_dft = strategy.fft2(g_padded)

        # Perform the DFT using torch
        g_dft_torch = torch.fft.fftn(g_padded)

        # Get the magnitude
        magnitude_torch = torch.abs(g_dft_torch)
        magnitude_mp = torch.abs(g_dft)

        # Check if the results are the same
        self.assertTrue(np.allclose(magnitude_torch.numpy(), magnitude_mp.numpy(), atol=1e-6))


    def test_torch_mpfft_2d_random_nopad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)
        plane1 = Plane(params_plane0, bits=128)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

        # Create a random input
        g = torch.randn(1, plane0.Nx, plane0.Ny).to(torch.complex128)

        # Perform the DFT
        g_dft = strategy.fft2(g)

        # Perform the DFT using torch
        g_dft_torch = torch.fft.fftn(g)

        # Get the magnitude
        magnitude_torch = torch.abs(g_dft_torch)
        magnitude_mp = torch.abs(g_dft)

        # Check if the results are the same
        self.assertTrue(np.allclose(magnitude_torch.numpy(), magnitude_mp.numpy(), atol=1e-6))


    def test_torch_mpfft_2d_random_pad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)
        plane1 = Plane(params_plane0, bits=128)

        kwargs = {'padded': True}
        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Create a random input
        g_padded = torch.zeros(1, plane0.Nx*2, plane0.Ny*2).to(torch.complex128)
        g = torch.randn(1, plane0.Nx, plane0.Ny).to(torch.complex128)
        g_padded[0, plane0.Nx//2:plane0.Nx+plane0.Nx//2, plane0.Ny//2:plane0.Ny+plane0.Ny//2] = g

        # Perform the DFT
        g_dft = strategy.fft2(g_padded)

        # Perform the DFT using torch
        g_dft_torch = torch.fft.fftn(g_padded)

        # Get the magnitude
        magnitude_torch = torch.abs(g_dft_torch)
        magnitude_mp = torch.abs(g_dft)

        # Check if the results are the same
        self.assertTrue(np.allclose(magnitude_torch.numpy(), magnitude_mp.numpy(), atol=1e-6))

    def test_mpifft_1d_nice_nopad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane0)

        # Create an input signal with known frequencies in 1D
        x = plane0.x
        g = torch.sin(2*torch.pi*x / 1.e+3) + torch.sin(4*torch.pi*x / 1.e+3) + torch.sin(6*torch.pi*x / 1.e+3)
        g = g.unsqueeze(0).to(torch.complex128)

        # Perform the DFT
        g_dft = strategy.fft(g)

        # Perform the IDFT
        g_idft = strategy.ifft(g_dft)

        fig, ax = plt.subplots(1,2)
        fig.suptitle('1D IDFT no pad')
        ax[0].plot(g_idft.real.squeeze().numpy())
        ax[0].set_title('IDFT')
        ax[1].plot(g.real.squeeze().numpy())
        ax[1].set_title('Original')
        plt.show()

    def test_mpifft_1d_nice_pad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)

        kwargs = {'padded': True}
        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane0, kwargs)

        # Create an input signal with known frequencies in 1D
        x = plane0.x
        g = torch.sin(2*torch.pi*x / 1.e+3) + torch.sin(4*torch.pi*x / 1.e+3) + torch.sin(6*torch.pi*x / 1.e+3)
        g_padded = torch.zeros(1, plane0.Nx*2).to(torch.complex128)
        g_padded[0, plane0.Nx//2:plane0.Nx+plane0.Nx//2] = g.to(torch.complex128)

        # Perform the DFT
        g_dft = strategy.fft(g_padded)

        # Perform the IDFT
        g_idft = strategy.ifft(g_dft)

        # Crop the signals
        g_idft = g_idft[:, plane0.Nx//2:plane0.Nx+plane0.Nx//2]

        fig, ax = plt.subplots(1,2)
        fig.suptitle('1D IDFT padded')
        ax[0].plot(g_idft.real.squeeze().numpy())
        ax[0].set_title('IDFT')
        ax[1].plot(g.real.squeeze().numpy())
        ax[1].set_title('Original')
        plt.show()


    def test_mpifft_2d_nice_nopad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)
        plane1 = Plane(params_plane0, bits=128)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

        # Create an input signal with known frequencies in 2D
        xx, yy = plane0.xx, plane0.yy
        g = torch.sin(2*torch.pi*xx / 1.e+3) + torch.sin(4*torch.pi*yy / 1.e+3) + torch.sin(6*torch.pi*xx / 1.e+3).to(torch.complex128)

        # Perform the DFT
        g_dft = strategy.fft2(g)

        # Perform the IDFT
        g_idft = strategy.ifft2(g_dft)

        fig, ax = plt.subplots(1,2)
        fig.suptitle('2D IDFT no pad')
        ax[0].imshow(g_idft.real.squeeze().numpy())
        ax[0].set_title('IDFT')
        ax[1].imshow(g.real.squeeze().numpy())
        ax[1].set_title('Original')
        plt.show()


    def test_mpifft_2d_nice_pad(self):
        # Create a plane
        plane0 = Plane(params_plane0, bits=128)
        plane1 = Plane(params_plane0, bits=128)

        kwargs = {'padded': True}
        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Create an input signal with known frequencies in 2D
        xx, yy = plane0.xx, plane0.yy
        g = torch.sin(2*torch.pi*xx / 1.e+3) + torch.sin(4*torch.pi*yy / 1.e+3) + torch.sin(6*torch.pi*xx / 1.e+3).to(torch.complex128)
        g_padded = torch.zeros(1, plane0.Nx*2, plane0.Ny*2).to(torch.complex128)
        g_padded[0, plane0.Nx//2:plane0.Nx+plane0.Nx//2, plane0.Ny//2:plane0.Ny+plane0.Ny//2] = g
        
        # Perform the DFT
        g_dft = strategy.fft2(g_padded)

        # Perform the IDFT
        g_idft = strategy.ifft2(g_dft)

        # Crop the signals
        g_idft = g_idft[:,:,plane0.Nx//2:plane0.Nx+plane0.Nx//2, plane0.Ny//2:plane0.Ny+plane0.Ny//2]

        fig, ax = plt.subplots(1,2)
        fig.suptitle('2D IDFT padded')
        ax[0].imshow(g_idft.real.squeeze().numpy())
        ax[0].set_title('IDFT')
        ax[1].imshow(g.real.squeeze().numpy())
        ax[1].set_title('Original')
        plt.show()


    def test_dft_to_smaller(self):
        # Create an input plane
        plane0 = Plane(params_plane0, bits=128)

        # Create an output plane
        plane1 = Plane(params_plane6, bits=128)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

        # Create an input signal with known frequencies in 2D
        xx, yy = plane0.xx, plane0.yy
        g = torch.sin(2*torch.pi*xx / 1.e+3) + torch.sin(4*torch.pi*yy / 1.e+3) + torch.sin(6*torch.pi*xx / 1.e+3).to(torch.complex128)

        # Perform the DFT
        g_dft = strategy.fft2(g)

        # Perform the IDFT
        g_idft = strategy.ifft2(g_dft)

        fig, ax = plt.subplots(1,2)
        fig.suptitle('DFT to smaller plane')
        ax[0].imshow(g_idft.real.squeeze().numpy())
        ax[0].set_title('IDFT')
        ax[1].imshow(g.real.squeeze().numpy())
        ax[1].set_title('Original')
        plt.show()

    def test_dft_to_larger(self):
        # Create an input plane
        plane0 = Plane(params_plane6, bits=128)

        # Create an output plane
        plane1 = Plane(params_plane0, bits=128)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

        # Create an input signal with known frequencies in 2D
        xx, yy = plane0.xx, plane0.yy
        g = torch.sin(2*torch.pi*xx / 1.e+3) + torch.sin(4*torch.pi*yy / 1.e+3) + torch.sin(6*torch.pi*xx / 1.e+3).to(torch.complex128)

        # Perform the DFT
        g_dft = strategy.fft2(g)

        # Perform the IDFT
        g_idft = strategy.ifft2(g_dft)

        fig, ax = plt.subplots(1,2)
        fig.suptitle('DFT to larger plane')
        ax[0].imshow(g_idft.real.squeeze().numpy())
        ax[0].set_title('IDFT')
        ax[1].imshow(g.real.squeeze().numpy())
        ax[1].set_title('Original')
        plt.show()

    def test_cuda_mpfft(self):
        # Create an input plane
        plane0 = Plane(params_plane6, bits=128)

        # Create an output plane
        plane1 = Plane(params_plane0, bits=128)

        # Initialize the strategy
        strategy = MPFFTStrategy(plane0, plane1)

        strategy = strategy.to('cuda')

        self.assertTrue(strategy.device == torch.device('cuda', 0))

def suite_mpfft():
    suite = unittest.TestSuite()
    #suite.addTest(TestDFT('test_init_mpfft'))
    #suite.addTest(TestDFT('test_init_mpfft_different_ny'))
    #suite.addTest(TestDFT('test_init_mpfft_different_nx'))
    #suite.addTest(TestDFT('test_init_mpfft_different_nx_ny'))
    #suite.addTest(TestDFT('test_init_mpfft_different_x_size'))
    #suite.addTest(TestDFT('test_init_mpfft_different_y_size'))
    #suite.addTest(TestDFT('test_init_mpfft_different_x_y_size'))
    #suite.addTest(TestDFT('test_fx_fy_picking_nopad'))
    #suite.addTest(TestDFT('test_fx_fy_picking_pad'))
    #suite.addTest(TestDFT('test_dft_matrix_dtype'))
    #suite.addTest(TestDFT('test_np_torch_dft'))
    #suite.addTest(TestDFT('test_dft_matrix_precisions_lowtol'))
    #suite.addTest(TestDFT('test_torch_mpfft_1d_nice_nopad'))
    ##suite.addTest(TestDFT('test_torch_mpfft_1d_nice_pad'))
    #suite.addTest(TestDFT('test_torch_mpfft_1d_random_nopad'))
    ##suite.addTest(TestDFT('test_torch_mpfft_1d_random_pad'))
    #suite.addTest(TestDFT('test_torch_mpfft_2d_nice_nopad'))
    ##suite.addTest(TestDFT('test_torch_mpfft_2d_nice_pad'))
    #suite.addTest(TestDFT('test_torch_mpfft_2d_random_nopad'))
    ##suite.addTest(TestDFT('test_torch_mpfft_2d_random_pad'))
    #suite.addTest(TestDFT('test_mpifft_1d_nice_nopad'))
    #suite.addTest(TestDFT('test_mpifft_1d_nice_pad'))
    #suite.addTest(TestDFT('test_mpifft_2d_nice_nopad'))
    #suite.addTest(TestDFT('test_mpifft_2d_nice_pad'))
    #suite.addTest(TestDFT('test_dft_to_smaller'))
    #suite.addTest(TestDFT('test_dft_to_larger'))
    suite.addTest(TestDFT('test_cuda_mpfft'))

    return suite

if __name__ == '__main__':
    loguru.logger.remove()
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite_mpfft())
