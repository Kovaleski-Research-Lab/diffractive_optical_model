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
sys.path.append('../../') # To get the fft strategies
sys.path.append('../../../../') # To get the plane


params_plane0 = {
    'name': 'plane0',
    'center': [0,0,0],
    'size': [8.64, 8.64],
    'normal': [0,0,1],
    'Nx': 1080,
    'Ny': 1080,
    }

params_plane1 = {
    'name': 'plane1',
    'center': [0,0,762],
    'size': [8.64, 8.64],
    'normal': [0,0,1],
    'Nx': 1080,
    'Ny': 1080,
    }

params_plane2 = {
    'name': 'plane2',
    'center': [0,0,200],
    'size': [4.32, 4.32],
    'normal': [0,0,1],
    'Nx': 540,
    'Ny': 540,
    }

params_plane3 = {
        'name': 'plane3',
        'center': [0,0,200],
        'size': [10.8, 10.8],
        'normal': [0,0,1],
        'Nx': 1350,
        'Ny': 1350,
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
from asm_strategy import ASMStrategy
from rsc_strategy import RSCStrategy
from fft_strategies.mp_strategy import MPFFTStrategy
from fft_strategies.pytorch_strategy import PyTorchFFTStrategy

wavelength = 520.e-6 #mm


def center_crop(image, Nx, Ny):
    """
    Center crop an image to the desired size
    """
    image = image.squeeze()
    x, y = image.shape
    startx = x//2 - (Nx//2)
    starty = y//2 - (Ny//2)
    return image[startx:startx+Nx,starty:starty+Ny]

class TestPropagation(unittest.TestCase):
    def setup(self):
        pass
    def tearDown(self):
        pass

    def test_init_asm_nopad(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)
        
        kwargs = {'padded': False}

        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the ASM strategy
        asm_strategy = ASMStrategy(plane0, plane1, fft_strategy, wavelength)

        self.assertTrue(asm_strategy.transfer_function is not None)
        self.assertTrue(len(asm_strategy.transfer_function.shape) == 3)
        self.assertTrue(asm_strategy.transfer_function.squeeze().shape == fft_strategy.fxx.shape)

    def test_init_asm_pad(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)
        
        kwargs = {'padded': True}

        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the ASM strategy
        asm_strategy = ASMStrategy(plane0, plane1, fft_strategy, wavelength)

        self.assertTrue(asm_strategy.transfer_function is not None)
        self.assertTrue(len(asm_strategy.transfer_function.shape) == 3)
        self.assertTrue(asm_strategy.transfer_function.squeeze().shape == fft_strategy.fxx.shape)

    def test_init_rsc_nopad(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)

        kwargs = {'padded': False}

        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the RSC strategy
        rsc_strategy = RSCStrategy(plane0, plane1, fft_strategy, wavelength)

        self.assertTrue(rsc_strategy.transfer_function is not None)
        self.assertTrue(len(rsc_strategy.transfer_function.shape) == 3)
        self.assertTrue(rsc_strategy.transfer_function.squeeze().shape == fft_strategy.fxx.shape)

    def test_init_rsc_pad(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)

        kwargs = {'padded': True}

        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the RSC strategy
        rsc_strategy = RSCStrategy(plane0, plane1, fft_strategy, wavelength)

        self.assertTrue(rsc_strategy.transfer_function is not None)
        self.assertTrue(len(rsc_strategy.transfer_function.shape) == 3)
        self.assertTrue(rsc_strategy.transfer_function.squeeze().shape == fft_strategy.fxx.shape)

    def test_plot_asm_transfer_function(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)

        kwargs = {'padded': False}

        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the ASM strategy
        asm_strategy = ASMStrategy(plane0, plane1, fft_strategy, wavelength)

        transfer_function = asm_strategy.transfer_function.squeeze().detach().numpy()

        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(np.abs(transfer_function))
        ax[0].set_title('Magnitude')
        ax[0].axis('off')
        ax[1].imshow(np.angle(transfer_function))
        ax[1].set_title('Phase')
        plt.show()

    def test_plot_rsc_transfer_function(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)

        kwargs = {'padded': False}

        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the RSC strategy
        rsc_strategy = RSCStrategy(plane0, plane1, fft_strategy, wavelength)

        transfer_function = rsc_strategy.transfer_function.squeeze().detach().numpy()

        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(np.abs(transfer_function))
        ax[0].set_title('Magnitude')
        ax[0].axis('off')
        ax[1].imshow(np.angle(transfer_function))
        ax[1].set_title('Phase')
        plt.show()

    def test_propagate_asm_mpfft(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)

        kwargs = {'padded': True}

        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)
        #fft_strategy = PyTorchFFTStrategy(plane0, plane1, kwargs)

        # Initialize the ASM strategy
        asm_strategy = ASMStrategy(plane0, plane1, fft_strategy, wavelength)

        # Create a wavefront
        x = torch.linspace(-plane0.Lx, plane0.Lx, 2*plane0.Nx)
        y = torch.linspace(-plane0.Ly, plane0.Ly, 2*plane0.Ny)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        wavefront = torch.ones_like(xx)
        wavefront[(xx**2 + yy**2) > (0.6)**2] = 0
        #wavefront = torch.zeros_like(xx)
        #wavefront[(xx<0.2) * (yy<10) * (xx>-0.2) * (yy>-10)] = 1
        wavefront = wavefront.view(1,2*plane0.Nx,2*plane0.Ny)
        wavefront = wavefront.type(torch.complex64)

        # Propagate the wavefront
        output_wavefront = asm_strategy.propagate(wavefront)

        # Plot the input and output wavefronts 
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(np.abs(wavefront.squeeze().detach().numpy()))
        ax[0].set_title('Input wavefront')
        ax[0].axis('off')
        ax[1].imshow(np.abs(output_wavefront.squeeze().detach().numpy()))
        ax[1].set_title('Output wavefront')
        ax[1].axis('off')
        plt.show()

    def test_propagate_rsc_mpfft(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)

        kwargs = {'padded': True}

        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)
        #fft_strategy = PyTorchFFTStrategy(plane0, plane1, kwargs)

        # Initialize the RSC strategy
        rsc_strategy = RSCStrategy(plane0, plane1, fft_strategy, wavelength)

        # Create a wavefront
        Nx = 2*plane0.Nx
        Ny = 2*plane0.Ny
        x = torch.linspace(-plane0.Lx, plane0.Lx, Nx)
        y = torch.linspace(-plane0.Ly, plane0.Ly, Ny)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        wavefront = torch.ones_like(xx)
        wavefront[(xx**2 + yy**2) > (0.6)**2] = 0
        #wavefront = torch.zeros_like(xx)
        #wavefront[(xx<0.2) * (yy<10) * (xx>-0.2) * (yy>-10)] = 1

        wavefront = wavefront.view(1, Nx, Ny)
        wavefront = wavefront.type(torch.complex64)

        # Propagate the wavefront
        output_wavefront = rsc_strategy.propagate(wavefront)

        # Plot the input and output wavefronts 
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(np.abs(wavefront.squeeze().detach().numpy()))
        ax[0].set_title('Input wavefront')
        ax[0].axis('off')
        ax[1].imshow(np.abs(output_wavefront.squeeze().detach().numpy()))
        ax[1].set_title('Output wavefront')
        ax[1].axis('off')
        plt.show()

    def test_propagate_asm_toSmaller(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane2)

        kwargs = {'padded': True}
        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the ASM strategy
        asm_strategy = ASMStrategy(plane0, plane1, fft_strategy, wavelength)

        # Create a wavefront
        Nx = 2*plane0.Nx
        Ny = 2*plane0.Ny
        x = torch.linspace(-plane0.Lx, plane0.Lx, Nx)
        y = torch.linspace(-plane0.Ly, plane0.Ly, Ny)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        wavefront = torch.ones_like(xx)
        wavefront[(xx**2 + yy**2) > (0.6)**2] = 0
        wavefront = wavefront.view(1, Nx, Ny)
        wavefront = wavefront.type(torch.complex64)

        # Propagate the wavefront
        output_wavefront = asm_strategy.propagate(wavefront)

        # Plot the input and output wavefronts
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(np.abs(wavefront.squeeze().detach().numpy()))
        ax[0].set_title('Input wavefront')
        ax[0].axis('off')
        ax[1].imshow(np.abs(output_wavefront.squeeze().detach().numpy()))
        ax[1].set_title('Output wavefront')
        ax[1].axis('off')
        plt.show()

    def test_propagate_rsc_toSmaller(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane2)

        kwargs = {'padded': True}
        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the RSC strategy
        rsc_strategy = RSCStrategy(plane0, plane1, fft_strategy, wavelength)

        # Create a wavefront
        Nx = 2*plane0.Nx
        Ny = 2*plane0.Ny
        x = torch.linspace(-plane0.Lx, plane0.Lx, Nx)
        y = torch.linspace(-plane0.Ly, plane0.Ly, Ny)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        wavefront = torch.ones_like(xx)
        wavefront[(xx**2 + yy**2) > (0.6)**2] = 0
        wavefront = wavefront.view(1, Nx, Ny)
        wavefront = wavefront.type(torch.complex64)

        # Propagate the wavefront
        output_wavefront = rsc_strategy.propagate(wavefront)

        # Plot the input and output wavefronts
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(np.abs(wavefront.squeeze().detach().numpy()))
        ax[0].set_title('Input wavefront')
        ax[0].axis('off')
        ax[1].imshow(np.abs(output_wavefront.squeeze().detach().numpy()))
        ax[1].set_title('Output wavefront')
        ax[1].axis('off')
        plt.show()

    def test_propagate_asm_toLarger(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane3)

        kwargs = {'padded': True}
        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the ASM strategy
        asm_strategy = ASMStrategy(plane0, plane1, fft_strategy, wavelength)

        # Create a wavefront
        Nx = 2*plane0.Nx
        Ny = 2*plane0.Ny
        x = torch.linspace(-plane0.Lx, plane0.Lx, Nx)
        y = torch.linspace(-plane0.Ly, plane0.Ly, Ny)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        wavefront = torch.ones_like(xx)
        wavefront[(xx**2 + yy**2) > (0.6)**2] = 0
        wavefront = wavefront.view(1, Nx, Ny)
        wavefront = wavefront.type(torch.complex64)

        # Propagate the wavefront
        output_wavefront = asm_strategy.propagate(wavefront)

        # Plot the input and output wavefronts
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(np.abs(wavefront.squeeze().detach().numpy()))
        ax[0].set_title('Input wavefront')
        ax[0].axis('off')
        ax[1].imshow(np.abs(output_wavefront.squeeze().detach().numpy()))
        ax[1].set_title('Output wavefront')
        ax[1].axis('off')
        plt.show()

    def test_propagate_rsc_toLarger(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane3)

        kwargs = {'padded': True}
        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the RSC strategy
        rsc_strategy = RSCStrategy(plane0, plane1, fft_strategy, wavelength)

        # Create a wavefront
        Nx = 2*plane0.Nx
        Ny = 2*plane0.Ny
        x = torch.linspace(-plane0.Lx, plane0.Lx, Nx)
        y = torch.linspace(-plane0.Ly, plane0.Ly, Ny)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        wavefront = torch.ones_like(xx)
        wavefront[(xx**2 + yy**2) > (0.6)**2] = 0
        wavefront = wavefront.view(1, Nx, Ny)
        wavefront = wavefront.type(torch.complex64)

        # Propagate the wavefront
        output_wavefront = rsc_strategy.propagate(wavefront)

        # Plot the input and output wavefronts
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(np.abs(wavefront.squeeze().detach().numpy()))
        ax[0].set_title('Input wavefront')
        ax[0].axis('off')
        ax[1].imshow(np.abs(output_wavefront.squeeze().detach().numpy()))
        ax[1].set_title('Output wavefront')
        ax[1].axis('off')
        plt.show()

    def test_rsc_vs_asm(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)

        kwargs = {'padded': True}
        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the ASM strategy
        asm_strategy = ASMStrategy(plane0, plane1, fft_strategy, wavelength)

        # Initialize the RSC strategy
        rsc_strategy = RSCStrategy(plane0, plane1, fft_strategy, wavelength)

        # Create a wavefront
        Nx = 2*plane0.Nx
        Ny = 2*plane0.Ny
        x = torch.linspace(-plane0.Lx, plane0.Lx, Nx)
        y = torch.linspace(-plane0.Ly, plane0.Ly, Ny)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        wavefront = torch.ones_like(xx)
        wavefront[(xx**2 + yy**2) > (0.6)**2] = 0
        wavefront = wavefront.view(1, Nx, Ny)
        wavefront = wavefront.type(torch.complex64)

        # Propagate the wavefront
        output_wavefront_asm = asm_strategy.propagate(wavefront)
        output_wavefront_rsc = rsc_strategy.propagate(wavefront)

        image_asm = np.abs(output_wavefront_asm.squeeze().detach().numpy())
        image_rsc = np.abs(output_wavefront_rsc.squeeze().detach().numpy())
        difference = image_asm - image_rsc

        # Plot the input, rsc and asm images, and the difference between them
        fig,ax = plt.subplots(1,4,figsize=(15,5))
        ax[0].imshow(np.abs(wavefront.squeeze().detach().numpy()))
        ax[0].set_title('Input wavefront')
        ax[0].axis('off')
        ax[1].imshow(image_asm)
        ax[1].set_title('Output wavefront ASM')
        ax[1].axis('off')
        ax[2].imshow(image_rsc)
        ax[2].set_title('Output wavefront RSC')
        ax[2].axis('off')
        ax[3].imshow(difference)
        ax[3].set_title('Difference')
        ax[3].axis('off')
        plt.show()

    def test_cuda_asm(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)

        kwargs = {'padded': True}
        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the ASM strategy
        asm_strategy = ASMStrategy(plane0, plane1, fft_strategy, wavelength)
        asm_strategy = asm_strategy.to('cuda')

        self.assertTrue(asm_strategy.transfer_function.is_cuda)

    def test_cuda_rsc(self):
        # Create two planes for propagation
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)

        kwargs = {'padded': True}

        # Initialize the MPFFT strategy
        fft_strategy = MPFFTStrategy(plane0, plane1, kwargs)

        # Initialize the RSC strategy
        rsc_strategy = RSCStrategy(plane0, plane1, fft_strategy, wavelength)
        rsc_strategy = rsc_strategy.to('cuda')

        self.assertTrue(rsc_strategy.transfer_function.is_cuda)

def suite_propagation():
    suite = unittest.TestSuite()
    #suite.addTest(TestPropagation('test_init_asm_nopad'))
    #suite.addTest(TestPropagation('test_init_asm_pad'))
    #suite.addTest(TestPropagation('test_init_rsc_nopad'))
    #suite.addTest(TestPropagation('test_init_rsc_pad'))
    #suite.addTest(TestPropagation('test_plot_asm_transfer_function'))
    #suite.addTest(TestPropagation('test_plot_rsc_transfer_function'))
    #suite.addTest(TestPropagation('test_propagate_asm_mpfft'))
    #suite.addTest(TestPropagation('test_propagate_rsc_mpfft'))
    #suite.addTest(TestPropagation('test_propagate_asm_toSmaller'))
    #suite.addTest(TestPropagation('test_propagate_rsc_toSmaller'))
    #suite.addTest(TestPropagation('test_propagate_asm_toLarger'))
    #suite.addTest(TestPropagation('test_propagate_rsc_toLarger'))
    #suite.addTest(TestPropagation('test_rsc_vs_asm'))
    suite.addTest(TestPropagation('test_cuda_asm'))
    suite.addTest(TestPropagation('test_cuda_rsc'))

    return suite

if __name__ == '__main__':
    loguru.logger.remove()
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite_propagation())
