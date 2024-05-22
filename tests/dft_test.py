
import unittest
import os
import sys
import torch
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dft import dft_1d, dft_2d, dift_1d, dift_2d
from plane import Plane


class TestDFT(unittest.TestCase):
    def setup(self):
        pass
    def tearDown(self):
        pass

    #-------------------------------------------------------------------------
    # Testing basic functionality
    #-------------------------------------------------------------------------
    def test_linspaces(self):
        x_np = np.linspace(-1, 1, 1000)
        x_torch = torch.linspace(-1, 1, 1000)
        self.assertTrue(np.allclose(x_np, x_torch.numpy()) and np.allclose(x_torch.numpy(), x_np))

    def test_meshgrids(self):
        x = np.linspace(-1, 1, 1000)
        y = np.linspace(-1, 1, 1000)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        x_torch = torch.linspace(-1, 1, 1000)
        y_torch = torch.linspace(-1, 1, 1000)
        xx_torch, yy_torch = torch.meshgrid(x_torch, y_torch, indexing='ij')
        self.assertTrue(np.allclose(xx, xx_torch.numpy()) and np.allclose(yy, yy_torch.numpy()) and np.allclose(xx_torch.numpy(), xx) and np.allclose(yy_torch.numpy(), yy))

    def test_fftfreq(self):
        x = np.linspace(-1, 1, 1000)
        fx = np.fft.fftfreq(1000, np.diff(x)[0])
        x_torch = torch.linspace(-1, 1, 1000)
        fx_torch = torch.fft.fftfreq(1000, torch.diff(x_torch)[0])
        self.assertTrue(np.allclose(fx, fx_torch.numpy()) and np.allclose(fx_torch.numpy(), fx))

    def test_transpose_1d_numpy(self):
        x = np.linspace(-1, 1, 1000)
        g = np.sin(2 * np.pi * 10 * x)
        g = g.reshape(1, -1)
        self.assertTrue(np.allclose(g.T, g.transpose(1,0)) and np.allclose(g.transpose(1,0), g.T)) 

    def test_transpose_1d_torch(self):
        x = torch.linspace(-1, 1, 1000)
        g = torch.sin(2 * np.pi * 10 * x)
        g = g.reshape(1, -1)
        self.assertTrue(torch.allclose(g.T, g.transpose(1,0)) and torch.allclose(g.transpose(1,0), g.T))

    def test_transpose_2d_numpy(self):
        x = np.linspace(-1, 1, 1000)
        y = np.linspace(-1, 1, 1000)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        g = np.sin(2 * np.pi * 10 * xx)
        g += np.sin(2 * np.pi * 10 * yy)
        g = g.reshape(1, g.shape[0], g.shape[1])
        self.assertTrue(np.allclose(g.T, g.transpose(0,2,1)) and np.allclose(g.transpose(0,2,1), g.T))

    def test_transpose_2d_torch(self):
        x = torch.linspace(-1, 1, 1000)
        y = torch.linspace(-1, 1, 1000)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        g = torch.sin(2 * np.pi * 10 * xx)
        g += torch.sin(2 * np.pi * 10 * yy)
        g = g.reshape(1, g.shape[0], g.shape[1])
        self.assertTrue(torch.allclose(g.T, g.permute(0,2,1)) and torch.allclose(g.permute(0,2,1), g.T))

    def test_transpose_permute_2d_numpy_torch(self):
        x = np.linspace(-1, 1, 1000)
        y = np.linspace(-1, 1, 1000)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        g = np.sin(2 * np.pi * 10 * xx)
        g += np.sin(2 * np.pi * 10 * yy)
        g = g.reshape(1, g.shape[0], g.shape[1])
        g_torch = torch.tensor(g)
        self.assertTrue(np.allclose(g.transpose(0,2,1), g_torch.permute(0,2,1).numpy()) and np.allclose(g_torch.permute(0,2,1).numpy(), g.transpose(0,2,1)))

    def test_diff(self):
        x_np = np.linspace(-1, 1, 1000)
        x_torch = torch.linspace(-1, 1, 1000)
        self.assertTrue(np.allclose(np.diff(x_np)[0], torch.diff(x_torch)[0].numpy()) and np.allclose(torch.diff(x_torch)[0].numpy(), np.diff(x_np)[0]))

    def test_outer(self):
        x = np.linspace(-1, 1, 1000)
        dx = np.diff(x)[0]
        fx = np.fft.fftfreq(1000, np.diff(x)[0])
        m = np.arange(1000)
        outer_np = np.outer(fx, m*dx)

        x_torch = torch.linspace(-1, 1, 1000)
        dx_torch = torch.diff(x_torch)[0]
        fx_torch = torch.fft.fftfreq(1000, torch.diff(x_torch)[0])
        m_torch = torch.arange(1000)
        outer_torch = torch.outer(fx_torch, m_torch*dx_torch)

        self.assertTrue(np.allclose(outer_np, outer_torch.numpy()) and np.allclose(outer_torch.numpy(), outer_np))
    
    def test_outer_shapes(self):
        x = np.linspace(-1, 1, 1000)
        dx = np.diff(x)[0]
        fx = np.fft.fftfreq(1000, np.diff(x)[0])
        m = np.arange(1000)
        outer_np = np.outer(fx, m*dx)

        x_torch = torch.linspace(-1, 1, 1000)
        dx_torch = torch.diff(x_torch)[0]
        fx_torch = torch.fft.fftfreq(1000, torch.diff(x_torch)[0])
        m_torch = torch.arange(1000)
        outer_torch = torch.outer(fx_torch, m_torch*dx_torch)

        self.assertTrue(outer_np.shape == outer_torch.shape)

    def test_pi(self):
        self.assertTrue(np.pi == torch.pi)

    def test_basic_complex_init(self):
        x = np.linspace(-1, 1, 1000)
        x_torch = torch.linspace(-1, 1, 1000)

        g_np = np.exp(1j * 2 * np.pi * 10 * x)
        g_torch = torch.exp(1j * 2 * torch.pi * 10 * x_torch)

        self.assertTrue(np.allclose(g_np, g_torch.numpy()) and np.allclose(g_torch.numpy(), g_np))

    def test_complex_dtypes(self):
        x = np.linspace(-1, 1, 1000)
        x_torch = torch.linspace(-1, 1, 1000)

        g_np = np.exp(1j * 2 * np.pi * 10 * x).astype(np.complex128)
        g_torch = torch.exp(1j * 2 * torch.pi * 10 * x_torch).type(torch.complex128)

        self.assertTrue(g_np.dtype == np.complex128 and g_torch.dtype == torch.complex128)
    
    def test_arange(self):
        m = np.arange(1000)
        m_torch = torch.arange(1000)
        self.assertTrue(np.allclose(m, m_torch.numpy()) and np.allclose(m_torch.numpy(), m))

    def test_np_exp_to_torch_exp(self):
        x = np.linspace(-1, 1, 1000)
        x_torch = torch.linspace(-1, 1, 1000)

        g_np = np.exp(2 * np.pi * 10 * x)
        g_torch = torch.exp(2 * torch.pi * 10 * x_torch)

        self.assertTrue(np.allclose(g_np, g_torch.numpy()) and np.allclose(g_torch.numpy(), g_np))

    def test_complex_numbers(self):
        x = np.linspace(-1, 1, 1000)
        x_torch = torch.linspace(-1, 1, 1000)

        g_np = np.exp(1j * 2 * np.pi * 10 * x)
        g_torch = torch.exp(1j * 2 * torch.pi * 10 * x_torch)

        self.assertTrue(np.allclose(g_np, g_torch.numpy()) and np.allclose(g_torch.numpy(), g_np))

    def test_complex_init_shapes(self):
        x = np.linspace(-1, 1, 1000)
        x_torch = torch.linspace(-1, 1, 1000)

        g_np = np.exp(1j * 2 * np.pi * 10 * x)
        g_torch = torch.exp(1j * 2 * torch.pi * 10 * x_torch)

        self.assertTrue(g_np.shape == g_torch.shape)

    def test_complex_init_conversion(self):
        x = np.linspace(-1, 1, 1000)
        dx = np.diff(x)[0]
        fx = np.fft.fftfreq(1000, np.diff(x)[0])
        m = np.arange(1000)

        outer_np = np.outer(fx, m*dx)
        g_np = np.exp(1j * 2 * np.pi * outer_np)
        g_torch = torch.tensor(g_np)

        self.assertTrue(np.allclose(g_np, g_torch.numpy()) and np.allclose(g_torch.numpy(), g_np))

    def test_exponential_argument(self):
        x = np.linspace(-1, 1, 1000)
        dx = np.diff(x)[0]
        fx = np.fft.fftfreq(1000, np.diff(x)[0])
        m = np.arange(1000)

        x_torch = torch.linspace(-1, 1, 1000)
        dx_torch = torch.diff(x_torch)[0]
        fx_torch = torch.fft.fftfreq(1000, torch.diff(x_torch)[0])
        m_torch = torch.arange(1000)

        outer_np = np.outer(fx, m*dx)
        outer_torch = torch.outer(fx_torch, m_torch*dx_torch)

        np_arg = 1j * 2 * np.pi * outer_np
        torch_arg = 1j * 2 * torch.pi * outer_torch

        self.assertTrue(np.allclose(np_arg, torch_arg.numpy()) and np.allclose(torch_arg.numpy(), np_arg))


    def test_complex_init(self):
        x = np.linspace(-1, 1, 1000, dtype=np.complex128)
        dx = np.diff(x)[0]
        fx = np.fft.fftfreq(1000, np.diff(x)[0])
        m = np.arange(1000)

        x_torch = torch.linspace(-1, 1, 1000, dtype=torch.complex128)
        dx_torch = torch.diff(x_torch)[0]
        fx_torch = torch.tensor(fx)
        m_torch = torch.arange(1000)

        outer_np = np.outer(fx, m*dx)
        outer_torch = torch.outer(fx_torch, m_torch*dx_torch)

        np_arg = 1j * 2 * np.pi * outer_np
        torch_arg = 1j * 2 * torch.pi * outer_torch

        g_np = np.exp(np_arg)
        g_torch = torch.exp(torch_arg)

        self.assertTrue(np.allclose(g_torch.numpy(), g_np) and np.allclose(g_np, g_torch.numpy()))


    #-------------------------------------------------------------------------
    # DFT Matrix initializations
    #-------------------------------------------------------------------------
    def test_dft_matrix_1d(self):
        x = np.linspace(-1, 1, 1000, dtype=np.complex128)
        fx = np.fft.fftfreq(1000, np.diff(x)[0])
        dx = np.diff(x)[0]
        M = len(x)
        m = np.arange(M)
        outer_np = np.outer(fx, m*dx)
        dft_matrix_np = np.exp(-2j * np.pi * outer_np)

        x = torch.linspace(-1, 1, 1000, dtype=torch.complex128)
        fx = torch.tensor(fx)
        dx = torch.diff(x)[0]
        M = len(x)
        m = torch.arange(M)
        outer_torch = torch.outer(fx, m*dx)
        dft_matrix_torch = torch.exp(-2j * torch.pi * outer_torch)

        self.assertTrue(np.allclose(dft_matrix_np, dft_matrix_torch.numpy()))

    def test_dft_matrix_2d(self):
        y = np.linspace(-1, 1, 1000, dtype=np.complex128)
        x = np.linspace(-1, 1, 1000, dtype=np.complex128)
        fx = np.fft.fftfreq(1000, np.diff(x)[0])
        fy = np.fft.fftfreq(1000, np.diff(y)[0])
        dx = np.diff(x)[0]
        dy = np.diff(y)[0]
        M = len(x)
        N = len(y)
        m = np.arange(M)
        n = np.arange(N)
        dft_matrix_x = np.exp(-2j * np.pi * np.outer(fx, m*dx))
        dft_matrix_y = np.exp(-2j * np.pi * np.outer(fy, n*dy))

        x = torch.linspace(-1, 1, 1000, dtype=torch.complex128)
        y = torch.linspace(-1, 1, 1000, dtype=torch.complex128)
        fx = torch.tensor(fx)
        fy = torch.tensor(fy)
        dx = torch.diff(x)[0]
        dy = torch.diff(y)[0]
        M = len(x)
        N = len(y)
        m = torch.arange(M)
        n = torch.arange(N)
        dft_matrix_x_torch = torch.exp(-2j * torch.pi * torch.outer(fx, m*dx))
        dft_matrix_y_torch = torch.exp(-2j * torch.pi * torch.outer(fy, n*dy))

        self.assertTrue(np.allclose(dft_matrix_x, dft_matrix_x_torch.numpy()) and np.allclose(dft_matrix_y, dft_matrix_y_torch.numpy()))

    def test_dft_matrix_from_plane(self):

        input_plane_params = {
            'name': 'input_plane',
            'size': torch.tensor([8.96e-3, 8.96e-3]),
            'Nx': 1000,
            'Ny': 1000,
            'normal': torch.tensor([0,0,1]),
            'center': torch.tensor([0,0,0])
        }
        input_plane = Plane(input_plane_params)

        x = input_plane.x
        y = input_plane.y
        M = len(x)
        N = len(y)
        m = torch.arange(M)
        n = torch.arange(N)
        dft_matrix_x_plane = torch.exp(-2j * torch.pi * torch.outer(input_plane.fx, m*input_plane.delta_x))
        dft_matrix_y_plane = torch.exp(-2j * torch.pi * torch.outer(input_plane.fy, n*input_plane.delta_y))

        y = np.linspace(-1, 1, 1000, dtype=np.complex128)
        x = np.linspace(-1, 1, 1000, dtype=np.complex128)
        fx = np.fft.fftfreq(1000, np.diff(x)[0])
        fy = np.fft.fftfreq(1000, np.diff(y)[0])
        dx = np.diff(x)[0]
        dy = np.diff(y)[0]
        M = len(x)
        N = len(y)
        m = np.arange(M)
        n = np.arange(N)
        dft_matrix_x = np.exp(-2j * np.pi * np.outer(fx, m*dx))
        dft_matrix_y = np.exp(-2j * np.pi * np.outer(fy, n*dy))

        self.assertTrue(np.allclose(dft_matrix_x, dft_matrix_x_plane.numpy()) and np.allclose(dft_matrix_y, dft_matrix_y_plane.numpy()))

    #-------------------------------------------------------------------------
    # 1D torch to numpy
    #-------------------------------------------------------------------------
    def test_external_dft_1d_torch_numpy(self):
        M = 1000
        x = torch.linspace(-1, 1, M, dtype=torch.complex128)
        fx = np.fft.fftfreq(M, np.diff(x)[0]) 
        fx = torch.tensor(fx)
        g = torch.sin(2 * np.pi * 10 * x)
        G_me = dft_1d(g, x, fx, backend=torch)
        G_np = np.fft.fft(g.numpy())
        self.assertTrue(np.allclose(G_me.numpy(), G_np))

    def test_external_dift_1d_torch_numpy(self):
        M = 1000
        x = torch.linspace(-1, 1, M, dtype=torch.complex128)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fx = torch.tensor(fx)
        g = torch.sin(2 * np.pi * 10 * x)
        G_me = dift_1d(g, x, fx, x, backend=torch)
        G_np = np.fft.ifft(g.numpy())
        self.assertTrue(np.allclose(G_me.numpy(), G_np))

    #-------------------------------------------------------------------------
    # 1D numpy to torch
    #-------------------------------------------------------------------------
    def test_external_dft_1d_numpy_torch(self):
        M = 1000
        x = np.linspace(-1, 1, M, dtype=np.complex128)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        g = np.sin(2 * np.pi * 10 * x)
        G_me = dft_1d(g, x, fx, backend=np)
        G_to = torch.fft.fft(torch.tensor(g).type(torch.complex128))
        self.assertTrue(np.allclose(G_me, G_to.numpy()))
    def test_external_dift_1d_numpy_torch(self):
        M = 1000
        x = np.linspace(-1, 1, M, dtype=np.complex128)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        g = np.sin(2 * np.pi * 10 * x)
        G_me = dift_1d(g, x, fx, x, backend=np)
        G_to = torch.fft.ifft(torch.tensor(g).type(torch.complex128))
        self.assertTrue(np.allclose(G_me, G_to.numpy()))

    #-------------------------------------------------------------------------
    # 1D torch to torch
    #-------------------------------------------------------------------------
    def test_external_dft_1d_torch_torch(self):
        M = 1000
        x = torch.linspace(-1, 1, M, dtype=torch.complex128)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fx = torch.tensor(fx)
        g = torch.sin(2 * np.pi * 10 * x)
        G_me = dft_1d(g, x, fx, backend=torch)
        G_torch = torch.fft.fft(g)
        self.assertTrue(torch.allclose(G_me, G_torch))

    def test_external_dift_1d_torch_torch(self):
        M = 1000
        x = torch.linspace(-1, 1, M, dtype=torch.complex128)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fx = torch.tensor(fx)
        g = torch.sin(2 * np.pi * 10 * x)
        G_me = dift_1d(g, x, fx, x, backend=torch)
        G_torch = torch.fft.ifft(g)
        self.assertTrue(torch.allclose(G_me, G_torch))

    #-------------------------------------------------------------------------
    # 1D numpy to numpy
    #-------------------------------------------------------------------------
    def test_external_dft_1d_numpy_numpy(self):
        M = 1000
        x = np.linspace(-1, 1, M, dtype=np.complex128)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        g = np.sin(2 * np.pi * 10 * x)
        G_me = dft_1d(g, x, fx, backend=np)
        G_np = np.fft.fft(g)
        self.assertTrue(np.allclose(G_me, G_np))

    def test_external_dift_1d_numpy_numpy(self):
        M = 1000
        x = np.linspace(-1, 1, M, dtype=np.complex128)
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        g = np.sin(2 * np.pi * 10 * x)
        G_me = dift_1d(g, x, fx, x, backend=np)
        G_np = np.fft.ifft(g)
        self.assertTrue(np.allclose(G_me, G_np))

    #-------------------------------------------------------------------------
    # 2D torch to numpy
    #-------------------------------------------------------------------------
    def test_external_dft_2d_torch_numpy(self):
        M = 1000
        N = 1000
        x = np.linspace(-1, 1, M, dtype=np.complex128)
        y = np.linspace(-1, 1, N, dtype=np.complex128)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        x = torch.tensor(x)
        y = torch.tensor(y)
        xx = torch.tensor(xx)
        yy = torch.tensor(yy)
        fx = torch.tensor(fx)
        fy = torch.tensor(fy)
        g = torch.sin(2 * np.pi * 10 * xx)
        g += torch.sin(2 * np.pi * 10 * yy)
        G_me = dft_2d(g.cuda(), x, y, fx, fy, backend=torch).cpu().squeeze().numpy()
        G_np = np.fft.fft2(g.cpu().numpy())
        self.assertTrue(np.allclose(G_me, G_np, atol=1e-7))

    def test_external_dift_2d_torch_numpy(self):
        M = 1000
        N = 1000
        x = np.linspace(-1, 1, M, dtype=np.complex128)
        y = np.linspace(-1, 1, N, dtype=np.complex128)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        x = torch.tensor(x)
        y = torch.tensor(y)
        xx = torch.tensor(xx)
        yy = torch.tensor(yy)
        fx = torch.tensor(fx)
        fy = torch.tensor(fy)
        g = torch.sin(2 * np.pi * 10 * xx)
        g += torch.sin(2 * np.pi * 10 * yy)
        G_me = dift_2d(g.cuda(), x, y, fx, fy, x, y, backend=torch).cpu().squeeze().numpy()
        G_np = np.fft.ifft2(g.numpy())
        self.assertTrue(np.allclose(G_me, G_np, atol=1e-7))
    #-------------------------------------------------------------------------
    # 2D numpy to torch
    #-------------------------------------------------------------------------
    def test_external_dft_2d_numpy_torch(self):
        M = 1000
        N = 1000
        x = np.linspace(-1, 1, M, dtype=np.complex128)
        y = np.linspace(-1, 1, N, dtype=np.complex128)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        fx = torch.tensor(fx)
        fy = torch.tensor(fy)
        g = np.sin(2 * np.pi * 10 * xx)
        g += np.sin(2 * np.pi * 10 * yy)
        G_me = dft_2d(g, x, y, fx, fy, backend=np).squeeze()
        G_to = torch.fft.fftn(torch.tensor(g).type(torch.complex128))
        self.assertTrue(np.allclose(G_me, G_to.numpy(), atol=1e-7))
    def test_external_dift_2d_numpy_torch(self):
        M = 1000
        N = 1000
        x = np.linspace(-1, 1, M, dtype=np.complex128)
        y = np.linspace(-1, 1, N, dtype=np.complex128)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        g = np.sin(2 * np.pi * 10 * xx)
        g += np.sin(2 * np.pi * 10 * yy)
        G_me = dift_2d(g, x, y, fx, fy, x, y, backend=np).squeeze()
        G_to = torch.fft.ifftn(torch.tensor(g))
        self.assertTrue(np.allclose(G_me, G_to.numpy(), atol=1e-7))
    #-------------------------------------------------------------------------
    # 2D torch to torch
    #-------------------------------------------------------------------------
    def test_external_dft_2d_torch_torch(self):
        M = 1000
        N = 1000

        x = np.linspace(-1, 1, M, dtype=np.complex128)
        y = np.linspace(-1, 1, N, dtype=np.complex128)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        x = torch.tensor(x)
        y = torch.tensor(y)
        xx = torch.tensor(xx)
        yy = torch.tensor(yy)
        fx = torch.tensor(fx)
        fy = torch.tensor(fy)
        g = torch.sin(2 * np.pi * 10 * xx)
        g += torch.sin(2 * np.pi * 10 * yy)
        G_me = dft_2d(g.cuda(), x, y, fx, fy, backend=torch).cpu()
        G_torch = torch.fft.fft2(g)
        self.assertTrue(torch.allclose(G_me, G_torch, atol=1e-7))
    def test_external_dift_2d_torch_torch(self):
        M = 1000
        N = 1000

        x = np.linspace(-1, 1, M, dtype=np.complex128)
        y = np.linspace(-1, 1, N, dtype=np.complex128)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        x = torch.tensor(x)
        y = torch.tensor(y)
        xx = torch.tensor(xx)
        yy = torch.tensor(yy)
        fx = torch.tensor(fx)
        fy = torch.tensor(fy)
        g = torch.sin(2 * np.pi * 10 * xx)
        g += torch.sin(2 * np.pi * 10 * yy)
        G_me = dift_2d(g.cuda(), x, y, fx, fy, x, y, backend=torch).cpu()
        G_torch = torch.fft.ifft2(g)
        self.assertTrue(torch.allclose(G_me, G_torch, atol=1e-7))
    #-------------------------------------------------------------------------
    # 2D numpy to numpy
    #-------------------------------------------------------------------------
    def test_external_dft_2d_numpy_numpy(self):
        M = 1000
        N = 1000
        x = np.linspace(-1, 1, M, dtype=np.complex128) 
        y = np.linspace(-1, 1, N, dtype=np.complex128)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        g = np.sin(2 * np.pi * 10 * xx)
        g += np.sin(2 * np.pi * 10 * yy)
        G_me = dft_2d(g, x, y, fx, fy, backend=np).squeeze()
        G_np = np.fft.fft2(g)
        self.assertTrue(np.allclose(G_me, G_np, atol=1e-7))
    def test_external_dift_2d_numpy_numpy(self):
        M = 1000
        N = 1000
        x = np.linspace(-1, 1, M, dtype=np.complex128)
        y = np.linspace(-1, 1, N, dtype=np.complex128)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        fx = np.fft.fftfreq(M, np.diff(x)[0])
        fy = np.fft.fftfreq(N, np.diff(y)[0])
        fx = np.fft.fftshift(fx)
        fy = np.fft.fftshift(fy)
        g = np.sin(2 * np.pi * 10 * xx)
        g += np.sin(2 * np.pi * 10 * yy)
        G_me = dift_2d(g, x, y, fx, fy, x, y, backend=np).squeeze()
        G_np = np.fft.ifft2(g)
        self.assertTrue(np.allclose(G_me, G_np, atol=1.e-7))


def suite_basic():
    suite = unittest.TestSuite()
    suite.addTest(TestDFT('test_linspaces'))
    suite.addTest(TestDFT('test_meshgrids'))
    suite.addTest(TestDFT('test_fftfreq'))
    suite.addTest(TestDFT('test_transpose_1d_numpy'))
    suite.addTest(TestDFT('test_transpose_1d_torch'))
    #suite.addTest(TestDFT('test_transpose_2d_numpy'))  
    #suite.addTest(TestDFT('test_transpose_2d_torch'))
    suite.addTest(TestDFT('test_transpose_permute_2d_numpy_torch'))
    suite.addTest(TestDFT('test_diff'))
    suite.addTest(TestDFT('test_outer'))
    suite.addTest(TestDFT('test_outer_shapes'))
    suite.addTest(TestDFT('test_pi'))
    suite.addTest(TestDFT('test_arange'))
    suite.addTest(TestDFT('test_np_exp_to_torch_exp'))
    suite.addTest(TestDFT('test_complex_numbers'))
    suite.addTest(TestDFT('test_complex_init_shapes'))
    suite.addTest(TestDFT('test_basic_complex_init'))
    suite.addTest(TestDFT('test_complex_dtypes'))
    suite.addTest(TestDFT('test_complex_init_conversion'))
    suite.addTest(TestDFT('test_exponential_argument'))
    suite.addTest(TestDFT('test_complex_init'))
    return suite

def suite_dft_matrices():
    suite = unittest.TestSuite()
    suite.addTest(TestDFT('test_dft_matrix_1d'))
    suite.addTest(TestDFT('test_dft_matrix_2d'))
    suite.addTest(TestDFT('test_dft_matrix_from_plane'))
    return suite

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

    ## 2D
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
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite_basic())
    runner.run(suite_dft_matrices())
    runner.run(suite_dft())
