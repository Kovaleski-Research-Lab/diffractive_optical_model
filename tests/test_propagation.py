import math

import torch
import pytest

from diffractive_optical_model.plane.plane import Plane
from diffractive_optical_model.propagator.factory import PropagatorFactory


WAVELENGTH = 0.00052  # mm (520 nm), matching the project unit system


def _plane(name='p', Nx=32, Ny=32, size=(1.0, 1.0), center=(0.0, 0.0, 0.0)):
    return Plane({
        'name': name,
        'center': list(center),
        'size': list(size),
        'normal': [0, 0, 1],
        'Nx': Nx,
        'Ny': Ny,
    }, bits=64)


def _params(prop_type, padded, fft_type='auto', wavelength=WAVELENGTH):
    return {
        'prop_type': prop_type,
        'fft_type': fft_type,
        'wavelength': wavelength,
        'padded': padded,
    }


def _propagator(prop_type, padded=False, Nx=32, z=0.05, fft_type='auto',
                size=(1.0, 1.0), wavelength=WAVELENGTH, out_size=None, out_N=None,
                shift=(0.0, 0.0)):
    pin = _plane('in', Nx=Nx, Ny=Nx, size=size, center=(0.0, 0.0, 0.0))
    n_out = out_N if out_N is not None else Nx
    s_out = out_size if out_size is not None else size
    pout = _plane('out', Nx=n_out, Ny=n_out, size=s_out, center=(shift[0], shift[1], z))
    return PropagatorFactory()(pin, pout, _params(prop_type, padded, fft_type, wavelength)), pin, pout


def _ones_field(plane):
    nx, ny = int(plane.Nx), int(plane.Ny)
    return torch.ones(1, 1, nx, ny, dtype=plane.complex_type_torch)


def test_factory_same_plane_uses_pytorch_fft():
    pin = _plane('a')
    pout = _plane('b', center=(0, 0, 0.1))
    prop = PropagatorFactory()(pin, pout, _params('asm', False))
    assert prop.fft_strategy.__class__.__name__ == 'PyTorchFFTStrategy'


def test_factory_mismatched_uses_scaled_czt():
    pin = _plane('a', Nx=32, size=(1.0, 1.0))
    pout = _plane('b', Nx=16, size=(0.5, 0.5), center=(0, 0, 0.1))
    prop = PropagatorFactory()(pin, pout, _params('asm', True, fft_type='auto'))
    assert prop.fft_strategy.__class__.__name__ == 'CZTStrategy'


def test_asm_distance_rejects_negative_shift_inflation():
    factory = PropagatorFactory()
    pin = _plane('a', Nx=32, size=(1.0, 1.0))
    # Large |shift| should not increase allowed z.
    plus = _plane('b', Nx=32, size=(1.0, 1.0), center=(0.8, 0.0, 10.0))
    minus = _plane('c', Nx=32, size=(1.0, 1.0), center=(-0.8, 0.0, 10.0))
    params = _params('auto', False)
    assert factory.check_asm_distance(pin, plus, params) == factory.check_asm_distance(pin, minus, params)


def test_plane_wave_unpadded_asm():
    z = 0.02
    prop, pin, _ = _propagator('asm', padded=False, Nx=32, z=z)
    field = _ones_field(pin)
    out = prop(field)
    k = 2 * torch.pi / WAVELENGTH
    expected_phase = torch.exp(torch.tensor(1j * k * z, dtype=field.dtype))
    assert out.shape == field.shape
    assert torch.allclose(out.abs(), torch.ones_like(out.abs()), atol=1e-3)
    # Global phase matches exp(j k z); compare a center sample.
    cx = out.shape[-2] // 2
    ratio = out[0, 0, cx, cx] / expected_phase
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)


def test_evanescent_decay_not_dc():
    # Coarse grid / long λ so some FFT bins are evanescent: λ > 2 dx.
    wavelength = 0.4
    Nx = 32
    size = (2.0, 2.0)
    z = 0.3
    prop, pin, _ = _propagator(
        'asm', padded=False, Nx=Nx, z=z, size=size, wavelength=wavelength
    )
    fx = pin.fx
    cutoff = 1.0 / wavelength
    candidates = fx[fx.abs() > cutoff]
    assert candidates.numel() > 0
    f_evan = candidates[candidates.abs().argmax()]
    xx = pin.xx
    field = torch.exp(1j * 2 * torch.pi * f_evan * xx).to(pin.complex_type_torch).view(1, 1, Nx, Nx)
    out = prop(field)
    kz_arg = 1 - (wavelength * float(f_evan)) ** 2
    kz_abs = (2 * torch.pi / wavelength) * abs(kz_arg) ** 0.5
    expected_mag = torch.exp(torch.tensor(-kz_abs * z))
    mag = out.abs().mean()
    dc_mag = field.abs().mean()
    assert mag < 0.5 * dc_mag
    assert torch.isclose(mag, expected_mag.to(mag.dtype), rtol=0.25)


def test_batched_matches_stacked():
    prop, pin, _ = _propagator('asm', padded=True, Nx=24, z=0.02)
    torch.manual_seed(1)
    batch = torch.randn(4, 1, int(pin.Nx), int(pin.Ny), dtype=pin.complex_type_torch)
    out_b = prop(batch)
    stacked = torch.cat([prop(batch[i : i + 1]) for i in range(4)], dim=0)
    assert out_b.shape[0] == 4
    assert torch.allclose(out_b, stacked, atol=1e-4, rtol=1e-4)


def test_asm_dc_transfer_has_unit_magnitude():
    prop, _, _ = _propagator('asm', padded=False, Nx=32, z=0.05)
    H = prop.propagation_strategy.transfer_function
    # Unshifted FFT: DC at [0, 0]
    assert torch.isclose(H[0, 0, 0].abs(), torch.tensor(1.0, dtype=H.real.dtype), atol=1e-5)


def test_lateral_shift_moves_peak():
    Nx = 32
    z = 0.02
    shift = 0.15
    prop, pin, pout = _propagator('asm', padded=True, Nx=Nx, z=z, shift=(shift, 0.0))
    field = torch.zeros(1, 1, Nx, Nx, dtype=pin.complex_type_torch)
    field[..., Nx // 2, Nx // 2] = 1.0
    out = prop(field)
    peak = out.abs().squeeze()
    iy = torch.argmax(peak.max(dim=0).values).item()
    ix = torch.argmax(peak.max(dim=1).values).item()
    x_peak = float(pout.x[ix])
    # The output coordinates are local. A source at global x=0 appears at
    # local x=-center_out.
    assert abs(x_peak + shift) <= 2 * float(pout.delta_x)
    assert abs(float(pout.center_x) + x_peak) <= 2 * float(pout.delta_x)


def test_padded_flag_shapes():
    prop_p, pin, _ = _propagator('asm', padded=True, Nx=16, z=0.02)
    prop_u, _, _ = _propagator('asm', padded=False, Nx=16, z=0.02)
    field = _ones_field(pin)
    assert prop_p(field).shape == field.shape
    assert prop_u(field).shape == field.shape
    assert prop_p.padding is not None
    assert prop_u.padding is None


def test_asm_rsc_dni_agree_in_overlap_regime():
    # z large enough that the RSC kernel is sampled, small enough for ASM (JOSAA 401908).
    Nx = 16
    z = 300.0
    size = (1.0, 1.0)
    prop_asm, pin, _ = _propagator('asm', padded=True, Nx=Nx, z=z, size=size)
    prop_rsc, _, _ = _propagator('rsc', padded=True, Nx=Nx, z=z, size=size)
    prop_dni, _, _ = _propagator('dni', padded=True, Nx=Nx, z=z, size=size)
    xx, yy = pin.xx, pin.yy
    aperture = ((xx ** 2 + yy ** 2) <= (0.2) ** 2).to(pin.complex_type_torch)
    field = aperture.view(1, 1, Nx, Nx)
    out_asm = prop_asm(field)
    out_rsc = prop_rsc(field)
    out_dni = prop_dni(field)

    def rel_l2(a, b):
        corr = (a.conj() * b).sum()
        a = a * torch.exp(1j * corr.angle())
        return (a - b).abs().pow(2).sum().sqrt() / b.abs().pow(2).sum().sqrt().clamp(min=1e-12)

    err_rsc_dni = float(rel_l2(out_rsc, out_dni))
    err_asm_rsc = float(rel_l2(out_asm, out_rsc))
    assert err_rsc_dni < 0.05
    assert err_asm_rsc < 0.15


def test_mismatched_planes_shape_and_dni():
    Nx_in, Nx_out = 16, 8
    z = 40.0
    pin = _plane('in', Nx=Nx_in, Ny=Nx_in, size=(0.4, 0.4))
    pout = _plane(
        'out', Nx=Nx_out, Ny=Nx_out, size=(0.2, 0.2), center=(0.05, 0, z)
    )
    field = ((pin.xx ** 2 + pin.yy ** 2) < 0.08 ** 2).to(
        pin.complex_type_torch
    ).view(1, 1, Nx_in, Nx_in)
    prop = PropagatorFactory()(pin, pout, _params('asm', True))
    out = prop(field)
    assert out.shape == (1, 1, Nx_out, Nx_out)
    assert prop.fft_strategy.__class__.__name__ == 'CZTStrategy'
    out_rsc = PropagatorFactory()(pin, pout, _params('rsc', True))(field)
    params_dni = _params('dni', True)
    prop_dni = PropagatorFactory()(pin, pout, params_dni)
    out_dni = prop_dni(field)
    assert out_dni.shape == (1, 1, Nx_out, Nx_out)

    corr = (out.conj() * out_dni).sum()
    out = out * torch.exp(1j * corr.angle())
    error = torch.linalg.vector_norm(out - out_dni) / torch.linalg.vector_norm(out_dni)
    assert float(error) < 0.2
    corr_rsc = (out_rsc.conj() * out_dni).sum()
    out_rsc = out_rsc * torch.exp(1j * corr_rsc.angle())
    rsc_error = torch.linalg.vector_norm(out_rsc - out_dni) / torch.linalg.vector_norm(out_dni)
    assert float(rsc_error) < 0.02


def test_mismatched_zero_distance_czt_preserves_constant_gain():
    pin = _plane('in', Nx=16, Ny=16, size=(0.4, 0.4))
    pout = _plane('out', Nx=8, Ny=8, size=(0.2, 0.2))
    prop = PropagatorFactory()(pin, pout, _params('asm', True))
    field = torch.ones(1, 1, 16, 16, dtype=torch.complex64)
    out = prop(field)
    assert out.shape == (1, 1, 8, 8)
    assert torch.allclose(out, torch.ones_like(out), atol=2e-4, rtol=2e-4)


def test_forced_pytorch_rejects_mismatched_grids():
    pin = _plane('in', Nx=16, Ny=16, size=(0.4, 0.4))
    pout = _plane('out', Nx=8, Ny=8, size=(0.2, 0.2), center=(0, 0, 1))
    with pytest.raises(ValueError, match='identical'):
        PropagatorFactory()(pin, pout, _params('asm', True, fft_type='pytorch'))


@pytest.mark.parametrize('prop_type', ['asm', 'rsc', 'dni'])
def test_zero_distance_is_identity(prop_type):
    prop, pin, _ = _propagator(prop_type, padded=True, Nx=15, z=0.0)
    torch.manual_seed(4)
    field = torch.randn(2, 1, 15, 15, dtype=pin.complex_type_torch)
    assert torch.allclose(prop(field), field)


def test_odd_rectangular_padded_propagation_and_real_promotion():
    pin = _plane('in', Nx=15, Ny=17, size=(1.0, 0.8))
    pout = _plane('out', Nx=15, Ny=17, size=(1.0, 0.8), center=(0, 0, 0.02))
    prop = PropagatorFactory()(pin, pout, _params('asm', True))
    field = torch.ones(2, 1, 15, 17)
    out = prop(field)
    assert out.shape == (2, 1, 15, 17)
    assert out.dtype == torch.complex64
    assert torch.isfinite(out).all()


def test_negative_rsc_is_conjugate_for_real_input():
    z = 300.0
    prop_plus, pin, _ = _propagator('rsc', padded=True, Nx=16, z=z)
    prop_minus, _, _ = _propagator('rsc', padded=True, Nx=16, z=-z)
    h_plus = prop_plus.propagation_strategy.transfer_function
    h_minus = prop_minus.propagation_strategy.transfer_function
    assert torch.allclose(h_minus, h_plus.conj(), atol=1e-6, rtol=1e-6)
    field = ((pin.xx ** 2 + pin.yy ** 2) < 0.2 ** 2).float().view(1, 1, 16, 16)
    assert torch.allclose(prop_minus(field), prop_plus(field).conj(), atol=2e-5, rtol=2e-5)


def test_shifted_asm_rsc_dni_agree_in_global_coordinates():
    Nx = 16
    z = 300.0
    shift = 0.125
    props = {
        kind: _propagator(kind, padded=True, Nx=Nx, z=z, shift=(shift, 0.0))[0]
        for kind in ('asm', 'rsc', 'dni')
    }
    pin = props['asm'].input_plane
    field = ((pin.xx ** 2 + pin.yy ** 2) < 0.18 ** 2).to(torch.complex64).view(1, 1, Nx, Nx)
    outputs = {kind: prop(field) for kind, prop in props.items()}

    def phase_aligned_error(a, b):
        corr = (a.conj() * b).sum()
        a = a * torch.exp(1j * corr.angle())
        return torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)

    assert float(phase_aligned_error(outputs['asm'], outputs['rsc'])) < 0.2
    assert float(phase_aligned_error(outputs['rsc'], outputs['dni'])) < 0.05


def test_rsc_sampling_guard_and_diagnostic_override():
    pin = _plane('in', Nx=32, size=(1.0, 1.0))
    pout = _plane('out', Nx=32, size=(1.0, 1.0), center=(0, 0, 0.02))
    params = _params('rsc', True)
    with pytest.raises(ValueError, match='adequately sampled'):
        PropagatorFactory()(pin, pout, params)
    params['allow_aliasing'] = True
    assert PropagatorFactory()(pin, pout, params).propagation_strategy.__class__.__name__ == 'RSCStrategy'


@pytest.mark.parametrize('prop_type,z', [('asm', 0.02), ('rsc', 600.0), ('dni', 300.0)])
def test_field_autograd_is_finite(prop_type, z):
    prop, pin, _ = _propagator(prop_type, padded=True, Nx=8, z=z)
    torch.manual_seed(5)
    field = torch.randn(1, 1, 8, 8, dtype=pin.complex_type_torch, requires_grad=True)
    loss = prop(field).abs().square().mean()
    loss.backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()


def test_asm_matches_paraxial_gaussian_beam_width():
    wavelength = WAVELENGTH
    z = 100.0
    waist = 0.1
    pin = _plane('in', Nx=128, Ny=128, size=(2.0, 2.0))
    pout = _plane('out', Nx=128, Ny=128, size=(2.0, 2.0), center=(0, 0, z))
    prop = PropagatorFactory()(pin, pout, _params('asm', True, wavelength=wavelength))
    field = torch.exp(-(pin.xx ** 2 + pin.yy ** 2) / waist ** 2)
    out = prop(field.view(1, 1, 128, 128))[0, 0]
    intensity = out.abs().square()
    mean_r2 = (
        intensity * (pout.xx ** 2 + pout.yy ** 2)
    ).sum() / intensity.sum()
    measured_waist = math.sqrt(2 * float(mean_r2))
    rayleigh_range = math.pi * waist ** 2 / wavelength
    expected_waist = waist * math.sqrt(1 + (z / rayleigh_range) ** 2)
    assert measured_waist == pytest.approx(expected_waist, rel=0.02)
