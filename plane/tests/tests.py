

import unittest
import sys
import numpy as np

sys.path.append('../')

# Create some plane paramters for the tests
from plane import Plane

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
    'size': [8.64, 8.64],
    'normal': [0,0,1],
    'Nx': 1080,
    'Ny': 1080,
    }


class TestPlane(unittest.TestCase):
    def setup(self):
        pass
    def tearDown(self):
        pass

    def test_plane_precisions(self):
        # Create a plane with 64 bits
        plane0 = Plane(params_plane1, bits=64)

        # Create a plane with 128 bits
        plane1 = Plane(params_plane1, bits=128)

        from IPython import embed; embed()

        # Check that x, y, xx, yy, fx, fy, fxx, fyy are the same between the two planes
        self.assertTrue(np.allclose(plane0.x.numpy(), plane1.x.numpy()))
        self.assertTrue(np.allclose(plane0.y.numpy(), plane1.y.numpy()))
        self.assertTrue(np.allclose(plane0.xx.numpy(), plane1.xx.numpy()))
        self.assertTrue(np.allclose(plane0.yy.numpy(), plane1.yy.numpy()))
        self.assertTrue(np.allclose(plane0.fx.numpy(), plane1.fx.numpy()))
        self.assertTrue(np.allclose(plane0.fy.numpy(), plane1.fy.numpy()))
        self.assertTrue(np.allclose(plane0.fxx.numpy(), plane1.fxx.numpy()))
        self.assertTrue(np.allclose(plane0.fyy.numpy(), plane1.fyy.numpy()))

def suite_plane():
    suite = unittest.TestSuite()
    suite.addTest(TestPlane('test_plane_precisions'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite_plane())





