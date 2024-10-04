

import unittest
import sys
import numpy as np
import torch

sys.path.append('../')

# Create some plane paramters for the tests
from diffractive_optical_model.plane.plane import Plane

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
    'Ny': 1080,
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
    'Nx': 1080,
    'Ny': 540,
    }

params_plane4 = {
    'name': 'plane4',
    'center': [0,0,0],
    'size': [4e+3, 4e+3],
    'normal': [0,0,1],
    'Nx': 1080,
    'Ny': 1080,
    'bits': 128,
    }

class TestPlane(unittest.TestCase):
    def setup(self):
        pass
    def tearDown(self):
        pass

    def test_plane_precisions(self):
        # Create a plane with 64 bits
        plane0 = Plane(params_plane1, bits=128)

        # Create a plane with 128 bits
        plane1 = Plane(params_plane1, bits=128)

        # Check that x, y, xx, yy, fx, fy, fxx, fyy are the same between the two planes
        self.assertTrue(np.allclose(plane0.x.numpy(), plane1.x.numpy()))
        self.assertTrue(np.allclose(plane0.y.numpy(), plane1.y.numpy()))
        self.assertTrue(np.allclose(plane0.xx.numpy(), plane1.xx.numpy()))
        self.assertTrue(np.allclose(plane0.yy.numpy(), plane1.yy.numpy()))
        self.assertTrue(np.allclose(plane0.fx.numpy(), plane1.fx.numpy()))
        self.assertTrue(np.allclose(plane0.fy.numpy(), plane1.fy.numpy()))
        self.assertTrue(np.allclose(plane0.fxx.numpy(), plane1.fxx.numpy()))
        self.assertTrue(np.allclose(plane0.fyy.numpy(), plane1.fyy.numpy()))

    def test_is_same_spatial(self):
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane1)
        plane2 = Plane(params_plane2)
        plane3 = Plane(params_plane3)

        self.assertTrue(plane0.is_same_spatial(plane1))
        self.assertFalse(plane0.is_same_spatial(plane2))
        self.assertFalse(plane0.is_same_spatial(plane3))

    def test_is_smaller(self):
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane4)

        self.assertFalse(plane0.is_smaller(plane1))
        self.assertTrue(plane1.is_smaller(plane0))

    def test_scale_notinplace(self):
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane0)

        scale = 0.6
        plane2 = plane0.scale(scale, inplace=False)

        # Make sure that the original planes are not modified
        self.assertTrue(plane0.is_same_spatial(plane1))

        # Make sure that the new plane is scaled correctly
        self.assertTrue(plane0.Lx*scale == plane2.Lx)
        self.assertTrue(plane0.Ly*scale == plane2.Ly)
        self.assertTrue(plane0.Nx == plane2.Nx)
        self.assertTrue(plane0.Ny == plane2.Ny)
        self.assertTrue((plane0.Lx*scale)/2 == torch.max(plane2.x))
        self.assertTrue((plane0.Ly*scale)/2 == torch.max(plane2.y))
        self.assertTrue(-(plane0.Lx*scale)/2 == torch.min(plane2.x))
        self.assertTrue(-(plane0.Ly*scale)/2 == torch.min(plane2.y))

    def test_scale_inplace(self):
        plane0 = Plane(params_plane0)
        plane1 = Plane(params_plane0)
        scale = 0.6
        plane2 = plane0.scale(scale, inplace=False)
        plane0.scale(scale, inplace=True)
        self.assertTrue(plane0.is_same_spatial(plane2))


def suite_plane():
    suite = unittest.TestSuite()
    suite.addTest(TestPlane('test_plane_precisions'))
    suite.addTest(TestPlane('test_is_same_spatial'))
    suite.addTest(TestPlane('test_is_smaller'))
    suite.addTest(TestPlane('test_scale_notinplace'))
    suite.addTest(TestPlane('test_scale_inplace'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite_plane())





