import torch
import pytest

from diffractive_optical_model.plane.plane import Plane
from diffractive_optical_model.propagator.strategies.fft_strategies.pytorch_strategy import PyTorchFFTStrategy
from diffractive_optical_model.propagator.strategies.fft_strategies.mp_strategy import MPFFTStrategy


def _plane(Nx=32, Ny=32, size=(1.0, 1.0), name='p'):
    return Plane({
        'name': name,
        'center': [0, 0, 0],
        'size': list(size),
        'normal': [0, 0, 1],
        'Nx': Nx,
        'Ny': Ny,
    }, bits=64)


def test_pytorch_fft2_last_two_dims_only():
    plane = _plane()
    fft = PyTorchFFTStrategy(plane, plane, {'padded': False})
    torch.manual_seed(0)
    field = torch.randn(3, 2, 32, 32, dtype=torch.complex64)
    transformed = fft.fft2(field)
    # Length-1 batch/channel FFT would change values if fftn were used on all axes.
    stacked = torch.stack([fft.fft2(field[i : i + 1]) for i in range(3)], dim=0).squeeze(1)
    assert torch.allclose(transformed, stacked, atol=1e-5)
    reconstructed = fft.ifft2(transformed)
    assert torch.allclose(reconstructed, field, atol=1e-5)


def test_pytorch_matches_manual_shift_pair():
    plane = _plane()
    fft = PyTorchFFTStrategy(plane, plane, {'padded': False})
    field = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    spec = fft.fft2(field)
    manual = torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)))
    assert torch.allclose(spec, manual, atol=1e-5)
    back = fft.ifft2(spec)
    manual_back = torch.fft.fftshift(torch.fft.ifft2(manual), dim=(-2, -1))
    assert torch.allclose(back, manual_back, atol=1e-5)


def test_yy_input_is_grid():
    plane = _plane()
    fft = PyTorchFFTStrategy(plane, plane, {'padded': False})
    assert torch.is_tensor(fft.yy_input)
    assert fft.yy_input.shape == plane.yy.shape
    fft_p = PyTorchFFTStrategy(plane, plane, {'padded': True})
    assert fft_p.yy_input.shape == plane.yy_padded.shape


def test_mpfft_matches_pytorch_same_size():
    plane = _plane(Nx=24, Ny=24)
    kwargs = {'padded': False}
    pt = PyTorchFFTStrategy(plane, plane, kwargs)
    mp = MPFFTStrategy(plane, plane, kwargs)
    field = torch.randn(2, 1, 24, 24, dtype=torch.complex64)
    spec_pt = pt.fft2(field)
    spec_mp = mp.fft2(field)
    assert spec_mp.shape == spec_pt.shape
    assert torch.allclose(spec_mp, spec_pt, atol=1e-4, rtol=1e-4)
    assert torch.allclose(mp.ifft2(spec_mp), pt.ifft2(spec_pt), atol=1e-4, rtol=1e-4)


def test_mpfft_padded_roundtrip():
    plane = _plane(Nx=16, Ny=16)
    mp = MPFFTStrategy(plane, plane, {'padded': True})
    field = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
    back = mp.ifft2(mp.fft2(field))
    assert torch.allclose(back, field, atol=1e-4, rtol=1e-4)


def test_mpfft_mismatched_shapes():
    pin = _plane(Nx=16, Ny=16, size=(1.0, 1.0), name='in')
    pout = _plane(Nx=8, Ny=8, size=(0.5, 0.5), name='out')
    mp = MPFFTStrategy(pin, pout, {'padded': False})
    field = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    spec = mp.fft2(field)
    recon = mp.ifft2(spec)
    assert recon.shape[-2] == 8
    assert recon.shape[-1] == 8


def test_pytorch_parseval_and_rectangular_roundtrip():
    plane = _plane(Nx=15, Ny=22, size=(1.2, 0.7))
    fft = PyTorchFFTStrategy(plane, plane, {'padded': False})
    torch.manual_seed(7)
    field = torch.randn(2, 3, 15, 22, dtype=torch.complex128)
    spectrum = fft.fft2(field)
    n = field.shape[-2] * field.shape[-1]
    assert torch.allclose(
        field.abs().square().sum(),
        spectrum.abs().square().sum() / n,
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(fft.ifft2(spectrum), field, atol=1e-10, rtol=1e-10)
