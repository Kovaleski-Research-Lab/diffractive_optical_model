import matplotlib.pyplot as plt
import unittest
import os
import sys
import torch
import numpy as np
import time
import loguru
import yaml

sys.path.append('../')
sys.path.append('../../')

from factory import PropagatorFactory
from plane.plane import Plane

wavelength = 520.e-6 #mm

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
    'center': [0,0,0],
    'size': [8.64, 8.64],
    'normal': [0,0,1],
    'Nx': 540,
    'Ny': 1080,
    }

params_plane2 = {
    'name': 'plane2',
    'center': [0,0,0],
    'size': [8.64, 8.64],
    'normal': [0,0,1],
    'Nx': 1080,
    'Ny': 540,
    }

params_plane3 = {
    'name': 'plane3',
    'center': [0,0,0],
    'size': [4.64, 8.64],
    'normal': [0,0,1],
    'Nx': 1080,
    'Ny': 1080,
    }

params_plane4 = {
    'name': 'plane4',
    'center': [0,0,0],
    'size': [8.64, 4.64],
    'normal': [0,0,1],
    'Nx': 1080,
    'Ny': 1080,
    }

def center_crop(image, Nx, Ny):
    """
    Center crop an image to the desired size
    """
    image = image.squeeze()
    x, y = image.shape
    startx = x//2 - (Nx//2)
    starty = y//2 - (Ny//2)
    return image[startx:startx+Nx,starty:starty+Ny]

class TestPropagationFactory(unittest.TestCase):
    def setup(self):
        pass
    def tearDown(self):
        pass

    def test_same_plane_init(self):
        # Create two planes with the same parameters
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane0)

        # Create a propagator
        params = {
            'prop_type': 'auto',
            'fft_type': 'auto',
            'wavelength': wavelength,
            }
        propagator = PropagatorFactory()(plane0, plane1, params)
        # Check that the fft strategy is the PyTorchFFTStrategy
        self.assertTrue(propagator.fft_strategy.__class__.__name__ == 'PyTorchFFTStrategy')

    def test_different_nx_ny(self):
        # Create two planes with different Nx and Ny from plane 0
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)
        plane2 = Plane(params_plane2)

        # Create a propagator
        params = {
            'prop_type': 'auto',
            'fft_type': 'auto',
            'wavelength': wavelength,
            }

        propagator0 = PropagatorFactory()(plane0, plane1, params)
        propagator1 = PropagatorFactory()(plane0, plane2, params)

        # Check that the fft strategy is the MPFFTStrategy
        self.assertTrue(propagator0.fft_strategy.__class__.__name__ == 'MPFFTStrategy')
        self.assertTrue(propagator1.fft_strategy.__class__.__name__ == 'MPFFTStrategy')

    def test_different_lx_ly(self):
        # Create two planes with different Lx and Ly from plane 0
        plane0 = Plane(params_plane0)
        plane3 = Plane(params_plane3)
        plane4 = Plane(params_plane4)

        # Create a propagator
        params = {
            'prop_type': 'auto',
            'fft_type': 'auto',
            'wavelength': wavelength,
            }

        propagator0 = PropagatorFactory()(plane0, plane3, params)
        propagator1 = PropagatorFactory()(plane0, plane4, params)

        # Check that the fft strategy is the MPFFTStrategy
        self.assertTrue(propagator0.fft_strategy.__class__.__name__ == 'MPFFTStrategy')
        self.assertTrue(propagator1.fft_strategy.__class__.__name__ == 'MPFFTStrategy')

    def test_different_nx_ly(self):
        # Using plane 1 and plane 4
        plane1 = Plane(params_plane1)
        plane4 = Plane(params_plane4)

        # Create a propagator
        params = {
            'prop_type': 'auto',
            'fft_type': 'auto',
            'wavelength': wavelength,
            }

        propagator = PropagatorFactory()(plane1, plane4, params)

        # Check that the fft strategy is the MPFFTStrategy
        self.assertTrue(propagator.fft_strategy.__class__.__name__ == 'MPFFTStrategy')

    def test_different_ny_lx(self):
        # Using plane 2 and plane 3
        plane2 = Plane(params_plane2)
        plane3 = Plane(params_plane3)

        # Create a propagator
        params = {
            'prop_type': 'auto',
            'fft_type': 'auto',
            'wavelength': wavelength,
            }

        propagator = PropagatorFactory()(plane2, plane3, params)

        # Check that the fft strategy is the MPFFTStrategy
        self.assertTrue(propagator.fft_strategy.__class__.__name__ == 'MPFFTStrategy')

def suite_propagationFactory():
    suite = unittest.TestSuite()
    suite.addTest(TestPropagationFactory('test_same_plane_init'))
    suite.addTest(TestPropagationFactory('test_different_nx_ny'))
    suite.addTest(TestPropagationFactory('test_different_lx_ly'))
    suite.addTest(TestPropagationFactory('test_different_nx_ly'))
    suite.addTest(TestPropagationFactory('test_different_ny_lx'))


    return suite

if __name__ == '__main__':
    loguru.logger.remove()
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite_propagationFactory())
