import pytest
import torch

from diffractive_optical_model.plane.plane import Plane
from diffractive_optical_model.propagator.strategies.fft_strategies.czt_strategy import (
    CZTFFTStrategy,
    CZTStrategy,
)


def _plane(nx, ny, size, bits=128, name="plane"):
    return Plane(
        {
            "name": name,
            "center": [0.13, -0.27, 0.0],
            "size": list(size),
            "normal": [0, 0, 1],
            "Nx": nx,
            "Ny": ny,
        },
        bits=bits,
    )


def _direct_sum(values, samples, evaluations, sign):
    phase = torch.exp(
        sign
        * 2j
        * torch.pi
        * evaluations[:, None]
        * samples[None, :]
    ).to(values.dtype)
    return values @ phase.transpose(-2, -1)


def _direct_forward_2d(values, strategy):
    phase_x = torch.exp(
        -2j * torch.pi * strategy.fx[:, None] * strategy.x_input[None, :]
    ).to(values.dtype)
    phase_y = torch.exp(
        -2j * torch.pi * strategy.fy[:, None] * strategy.y_input[None, :]
    ).to(values.dtype)
    return torch.einsum("kn,...nm,pm->...kp", phase_x, values, phase_y)


def _direct_inverse_2d(values, strategy):
    phase_x = torch.exp(
        2j * torch.pi * strategy.x_output[:, None] * strategy.fx[None, :]
    ).to(values.dtype)
    phase_y = torch.exp(
        2j * torch.pi * strategy.y_output[:, None] * strategy.fy[None, :]
    ).to(values.dtype)
    scale = (
        strategy.input_plane.delta_x
        * strategy.delta_fx
        * strategy.input_plane.delta_y
        * strategy.delta_fy
    )
    return (
        torch.einsum("nk,...kp,mp->...nm", phase_x, values, phase_y)
        * scale
    )


def test_frequency_grid_obeys_support_and_nyquist_constraints():
    pin = _plane(7, 5, (1.4, 0.75), name="in")
    pout = _plane(9, 4, (2.7, 1.6), name="out")
    czt = CZTStrategy(pin, pout)

    for frequencies, spacing, dx_in, dx_out, support_in, support_out in (
        (czt.fx, czt.delta_fx, pin.delta_x, pout.delta_x, pin.Lx, pout.Lx),
        (czt.fy, czt.delta_fy, pin.delta_y, pout.delta_y, pin.Ly, pout.Ly),
    ):
        assert torch.all(torch.diff(frequencies) > 0)
        nyquist = 0.5 / torch.maximum(dx_in, dx_out)
        assert frequencies.abs().max() <= nyquist
        assert spacing <= 1 / torch.maximum(support_in, support_out)
        bandwidth = frequencies.numel() * spacing
        assert torch.allclose(bandwidth, 1 / torch.maximum(dx_in, dx_out))


def test_1d_unequal_grid_matches_direct_forward_and_inverse_sums():
    pin = _plane(7, 3, (1.4, 0.6), name="in")
    pout = _plane(6, 3, (1.8, 0.6), name="out")
    czt = CZTStrategy(pin, pout)
    torch.manual_seed(4)
    values = torch.randn(2, 7, dtype=torch.complex128)

    actual_spectrum = czt.fft(values)
    expected_spectrum = _direct_sum(values, czt.x_input, czt.fx, sign=-1)
    assert actual_spectrum.shape == (2, czt.fx.numel())
    assert torch.allclose(actual_spectrum, expected_spectrum, atol=2e-12, rtol=2e-12)

    actual_output = czt.ifft(actual_spectrum)
    expected_output = _direct_sum(
        actual_spectrum, czt.fx, czt.x_output, sign=1
    ) * (pin.delta_x * czt.delta_fx)
    assert actual_output.shape == (2, 6)
    assert torch.allclose(actual_output, expected_output, atol=2e-12, rtol=2e-12)


def test_2d_unequal_rectangular_grids_match_direct_matrix_sums():
    pin = _plane(5, 4, (1.0, 0.72), name="in")
    pout = _plane(4, 6, (0.68, 1.5), name="out")
    czt = CZTStrategy(pin, pout)
    torch.manual_seed(8)
    values = torch.randn(2, 1, 5, 4, dtype=torch.complex128)

    spectrum = czt.fft2(values)
    expected_spectrum = _direct_forward_2d(values, czt)
    assert spectrum.shape[-2:] == (czt.fx.numel(), czt.fy.numel())
    assert torch.allclose(spectrum, expected_spectrum, atol=3e-12, rtol=3e-12)

    output = czt.ifft2(spectrum)
    expected_output = _direct_inverse_2d(spectrum, czt)
    assert output.shape[-2:] == (4, 6)
    assert torch.allclose(output, expected_output, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize(
    ("bits", "dtype", "atol"),
    [(64, torch.complex64, 3e-5), (128, torch.complex128, 3e-12)],
)
def test_same_grid_roundtrip_and_dtype(bits, dtype, atol):
    plane = _plane(7, 6, (1.4, 0.9), bits=bits)
    czt = CZTFFTStrategy(plane, plane)
    torch.manual_seed(1)
    values = torch.randn(3, 2, 7, 6, dtype=dtype)
    reconstructed = czt.ifft2(czt.fft2(values))
    assert reconstructed.dtype == dtype
    assert torch.allclose(reconstructed, values, atol=atol, rtol=atol)


def test_constant_and_single_fourier_mode_land_in_centered_bins():
    plane = _plane(8, 5, (2.0, 1.0))
    czt = CZTStrategy(plane, plane)

    constant = torch.ones(8, dtype=torch.complex128)
    constant_spectrum = czt.fft(constant)
    dc = czt.fx.numel() // 2
    expected = torch.zeros_like(constant_spectrum)
    expected[dc] = 8
    assert torch.allclose(constant_spectrum, expected, atol=2e-12, rtol=2e-12)

    mode_bin = dc + 2
    mode = torch.exp(2j * torch.pi * czt.fx[mode_bin] * czt.x_input)
    mode_spectrum = czt.fft(mode)
    expected.zero_()
    expected[mode_bin] = 8
    assert torch.allclose(mode_spectrum, expected, atol=2e-12, rtol=2e-12)


def test_padded_grids_match_direct_sums_and_expose_padded_coordinates():
    pin = _plane(3, 4, (0.9, 0.8), name="in")
    pout = _plane(5, 3, (1.25, 0.75), name="out")
    czt = CZTStrategy(pin, pout, {"padded": True})
    torch.manual_seed(3)
    values = torch.randn(1, 1, 6, 8, dtype=torch.complex128)

    assert czt.xx_input.shape == (6, 8)
    assert czt.xx_output.shape == (10, 6)
    spectrum = czt.fft2(values)
    assert torch.allclose(
        spectrum, _direct_forward_2d(values, czt), atol=4e-12, rtol=4e-12
    )
    output = czt.ifft2(spectrum)
    assert output.shape[-2:] == (10, 6)
    assert torch.allclose(
        output, _direct_inverse_2d(spectrum, czt), atol=4e-12, rtol=4e-12
    )


def test_fft2_is_autograd_compatible():
    plane = _plane(3, 4, (0.9, 1.2))
    czt = CZTStrategy(plane, plane)
    torch.manual_seed(9)
    values = torch.randn(1, 3, 4, dtype=torch.complex128, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda value: czt.ifft2(czt.fft2(value)),
        (values,),
        eps=1e-6,
        atol=2e-6,
        rtol=2e-5,
    )
    loss = czt.fft2(values).abs().square().sum()
    gradient = torch.autograd.grad(loss, values)[0]
    assert gradient.dtype == torch.complex128
    assert torch.isfinite(gradient.real).all()
    assert torch.isfinite(gradient.imag).all()
