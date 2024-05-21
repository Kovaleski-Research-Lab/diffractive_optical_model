
import unittest
import os
import sys
import torch
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dft import dft_1d, dft_2d, dift_1d, dift_2d


class TestDFT(unittest.TestCase):
    def setup(self):
        pass
    def tearDown(self):
        pass

    #-------------------------------------------------------------------------
    # Testing basic functionality
    #-------------------------------------------------------------------------
    def test_linspaces(self):


    #-------------------------------------------------------------------------
    # 1D torch to numpy
    #-------------------------------------------------------------------------
    def test_external_dft_1d_torch_numpy(self):
        M = 1000
        x = torch.linspace(-1, 1, M)
        fx = torch.fft.fftfreq(M, torch.diff(x)[0])
        g = torch.sin(2 * np.pi * 10 * x).type(torch.complex128)
        G_me = dft_1d(g, x, fx, backend=torch)
        G_np = np.fft.fft(g.numpy())
        self.assertTrue(np.allclose(G_me.numpy(), G_np))

    def test_external_dift_1d_torch_numpy(self):
        M = 1000
        x = torch.linspace(-1, 1, M)
        fx = torch.fft.fftfreq(M, torch.diff(x)[0])
        g = torch.sin(2 * np.pi * 10 * x).type(torch.complex128)
        G_me = dift_1d(g, x, fx, x, backend=torch)
        G_np = np.fft.ifft(g.numpy())
        self.assertTrue(np.allclose(G_me.numpy(), G_np, atol=1e-5))


    #-------------------------------------------------------------------------
    # 1D numpy to torch
    #-------------------------------------------------------------------------
    def test_external_dft_1d_numpy_torch(self):
        M = 1000
        x = np.linspace(-1, 1, M)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        g = np.sin(2 * np.pi * 10 * x)
        G_me = dft_1d(g, x, fx, backend=np)
        G_to = torch.fft.fft(torch.tensor(g).type(torch.complex128))
        self.assertTrue(np.allclose(G_me, G_to.numpy()))
    def test_external_dift_1d_numpy_torch(self):
        M = 1000
        x = np.linspace(-1, 1, M)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        g = np.sin(2 * np.pi * 10 * x)
        G_me = dift_1d(g, x, fx, x, backend=np)
        G_to = torch.fft.ifft(torch.tensor(g).type(torch.complex128))
        self.assertTrue(np.allclose(G_me, G_to.numpy(), atol=1e-5))

    #-------------------------------------------------------------------------
    # 1D torch to torch
    #-------------------------------------------------------------------------
    def test_external_dft_1d_torch_torch(self):
        M = 1000
        x = torch.linspace(-1, 1, M)
        fx = torch.fft.fftfreq(M, torch.diff(x)[0])
        g = torch.sin(2 * np.pi * 10 * x).type(torch.complex128)
        G_me = dft_1d(g, x, fx, backend=torch)
        G_torch = torch.fft.fft(g)
        self.assertTrue(torch.allclose(G_me, G_torch))

    def test_external_dift_1d_torch_torch(self):
        M = 1000
        x = torch.linspace(-1, 1, M)
        fx = torch.fft.fftfreq(M, torch.diff(x)[0])
        g = torch.sin(2 * np.pi * 10 * x).type(torch.complex128)
        G_me = dift_1d(g, x, fx, x, backend=torch)
        G_torch = torch.fft.ifft(g)
        self.assertTrue(torch.allclose(G_me, G_torch, atol=1e-5))

    #-------------------------------------------------------------------------
    # 1D numpy to numpy
    #-------------------------------------------------------------------------
    def test_external_dft_1d_numpy_numpy(self):
        M = 1000
        x = np.linspace(-1, 1, M)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        g = np.sin(2 * np.pi * 10 * x)
        G_me = dft_1d(g, x, fx, backend=np)
        G_np = np.fft.fft(g)
        self.assertTrue(np.allclose(G_me, G_np))

    def test_external_dift_1d_numpy_numpy(self):
        M = 1000
        x = np.linspace(-1, 1, M)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        g = np.sin(2 * np.pi * 10 * x)
        G_me = dift_1d(g, x, fx, x, backend=np)
        G_np = np.fft.ifft(g)
        self.assertTrue(np.allclose(G_me, G_np, atol=1e-5))

    #-------------------------------------------------------------------------
    # 2D torch to numpy
    #-------------------------------------------------------------------------
    def test_external_dft_2d_torch_numpy(self):
        M = 1000
        N = 1000
        x = torch.linspace(-1, 1, M)
        y = torch.linspace(-1, 1, N)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        fx = torch.fft.fftfreq(M, torch.diff(x)[0])
        fy = torch.fft.fftfreq(N, torch.diff(y)[0])
        g = torch.sin(2 * np.pi * 10 * xx).type(torch.complex128)
        g += torch.sin(2 * np.pi * 10 * yy).type(torch.complex128)
        G_me = dft_2d(g, x, y, fx, fy, backend=torch).squeeze().numpy()
        G_np = np.fft.fft2(g.numpy())
        self.assertTrue(np.allclose(G_me, G_np, atol=1e-3))
    def test_external_dift_2d_torch_numpy(self):
        M = 1000
        N = 1000
        x = torch.linspace(-1, 1, M)
        y = torch.linspace(-1, 1, N)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        fx = torch.fft.fftfreq(M, torch.diff(x)[0])
        fy = torch.fft.fftfreq(N, torch.diff(y)[0])
        g = torch.sin(2 * np.pi * 10 * xx).type(torch.complex128)
        g += torch.sin(2 * np.pi * 10 * yy).type(torch.complex128)
        G_me = dift_2d(g, x, y, fx, fy, x, y, backend=torch).squeeze().numpy()
        G_np = np.fft.ifft2(g.numpy())
        self.assertTrue(np.allclose(G_me, G_np, atol=1e-3))
    #-------------------------------------------------------------------------
    # 2D numpy to torch
    #-------------------------------------------------------------------------
    def test_external_dft_2d_numpy_torch(self):
        M = 1000
        N = 1000
        x = np.linspace(-1, 1, M)
        y = np.linspace(-1, 1, N)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        g = np.sin(2 * np.pi * 10 * xx)
        g += np.sin(2 * np.pi * 10 * yy)
        G_me = dft_2d(g, x, y, fx, fy, backend=np).squeeze()
        G_to = torch.fft.fftn(torch.tensor(g).type(torch.complex128))
        self.assertTrue(np.allclose(G_me, G_to.numpy(), atol=1e-3))
    def test_external_dift_2d_numpy_torch(self):
        M = 1000
        N = 1000
        x = np.linspace(-1, 1, M)
        y = np.linspace(-1, 1, N)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        g = np.sin(2 * np.pi * 10 * xx)
        g += np.sin(2 * np.pi * 10 * yy)
        G_me = dift_2d(g, x, y, fx, fy, x, y, backend=np).squeeze()
        G_to = torch.fft.ifftn(torch.tensor(g).type(torch.complex128))
        self.assertTrue(np.allclose(G_me, G_to.numpy(), atol=1e-3))
    #-------------------------------------------------------------------------
    # 2D torch to torch
    #-------------------------------------------------------------------------
    def test_external_dft_2d_torch_torch(self):
        M = 1000
        N = 1000
        x = torch.linspace(-1, 1, M)
        y = torch.linspace(-1, 1, N)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        fx = torch.fft.fftfreq(M, torch.diff(x)[0])
        fy = torch.fft.fftfreq(N, torch.diff(y)[0])
        g = torch.sin(2 * np.pi * 10 * xx).type(torch.complex128)
        g += torch.sin(2 * np.pi * 10 * yy).type(torch.complex128)
        G_me = dft_2d(g, x, y, fx, fy, backend=torch)
        G_torch = torch.fft.fft2(g)
        self.assertTrue(torch.allclose(G_me, G_torch, atol=1e-3))
    def test_external_dift_2d_torch_torch(self):
        M = 1000
        N = 1000
        x = torch.linspace(-1, 1, M)
        y = torch.linspace(-1, 1, N)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        fx = torch.fft.fftfreq(M, torch.diff(x)[0])
        fy = torch.fft.fftfreq(N, torch.diff(y)[0])
        g = torch.sin(2 * np.pi * 10 * xx).type(torch.complex128)
        g += torch.sin(2 * np.pi * 10 * yy).type(torch.complex128)
        G_me = dift_2d(g, x, y, fx, fy, x, y, backend=torch)
        G_torch = torch.fft.ifft2(g)
        self.assertTrue(torch.allclose(G_me, G_torch, atol=1e-3))
    #-------------------------------------------------------------------------
    # 2D numpy to numpy
    #-------------------------------------------------------------------------
    def test_external_dft_2d_numpy_numpy(self):
        M = 1000
        N = 1000
        x = np.linspace(-1, 1, M)
        y = np.linspace(-1, 1, N)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        g = np.sin(2 * np.pi * 10 * xx)
        g += np.sin(2 * np.pi * 10 * yy)
        G_me = dft_2d(g, x, y, fx, fy, backend=np).squeeze()
        G_np = np.fft.fft2(g)
        self.assertTrue(np.allclose(G_me, G_np, atol=1e-3))
    def test_external_dift_2d_numpy_numpy(self):
        M = 1000
        N = 1000
        x = np.linspace(-1, 1, M)
        y = np.linspace(-1, 1, N)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        g = np.sin(2 * np.pi * 10 * xx)
        g += np.sin(2 * np.pi * 10 * yy)
        G_me = dift_2d(g, x, y, fx, fy, x, y, backend=np).squeeze()
        G_np = np.fft.ifft2(g)
        self.assertTrue(np.allclose(G_me, G_np, atol=1e-3))



def suite_dft():
    suite = unittest.TestSuite()
    # 1D
    suite.addTest(TestDFT('test_external_dft_1d_torch_numpy'))
    suite.addTest(TestDFT('test_external_dift_1d_torch_numpy'))

    suite.addTest(TestDFT('test_external_dft_1d_numpy_torch'))
    suite.addTest(TestDFT('test_external_dift_1d_numpy_torch'))

    suite.addTest(TestDFT('test_external_dft_1d_torch_torch'))
    suite.addTest(TestDFT('test_external_dift_1d_torch_torch'))

    suite.addTest(TestDFT('test_external_dft_1d_numpy_numpy'))
    suite.addTest(TestDFT('test_external_dift_1d_numpy_numpy'))

    # 2D
    suite.addTest(TestDFT('test_external_dft_2d_torch_numpy'))
    suite.addTest(TestDFT('test_external_dift_2d_torch_numpy'))

    suite.addTest(TestDFT('test_external_dft_2d_torch_torch'))
    suite.addTest(TestDFT('test_external_dift_2d_torch_torch'))

    suite.addTest(TestDFT('test_external_dft_2d_numpy_torch'))
    suite.addTest(TestDFT('test_external_dift_2d_numpy_torch'))

    suite.addTest(TestDFT('test_external_dft_2d_numpy_numpy'))
    suite.addTest(TestDFT('test_external_dift_2d_numpy_numpy'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite_dft())
