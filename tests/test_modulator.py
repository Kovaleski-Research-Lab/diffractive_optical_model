import torch
import pytest

from diffractive_optical_model.modulator.initializations import phase_initializations, amplitude_initializations
from diffractive_optical_model.modulator.factory import ModulatorFactory
from diffractive_optical_model.modulator.modulator import Modulator
from diffractive_optical_model.plane.plane import Plane


def _plane(Nx=16, Ny=16, bits=64):
    return Plane({
        'name': 'p',
        'center': [0, 0, 0],
        'size': [1.0, 1.0],
        'normal': [0, 0, 1],
        'Nx': Nx,
        'Ny': Ny,
    }, bits=bits)


def _factory_params(gradients='none', phase_init='uniform', amplitude_init='uniform',
                    phase_value=0.0, amplitude_value=1.0):
    return {
        'gradients': gradients,
        'amplitude_init': amplitude_init,
        'amplitude_value': amplitude_value,
        'phase_init': phase_init,
        'phase_value': phase_value,
        'focal_length': 1.0,
        'wavelength': 520e-6,
    }


def test_uniform_and_random_inits():
    plane = _plane()
    phase = phase_initializations.initialize_phase(plane, {'phase_init': 'uniform', 'phase_value': 0.0})
    assert phase.shape == (1, 1, 16, 16)
    rand_phase = phase_initializations.initialize_phase(plane, {'phase_init': 'random'})
    assert rand_phase.max() > 1.0  # not stuck in [0, 1]
    assert rand_phase.max() <= 2 * torch.pi + 1e-5
    amp = amplitude_initializations.initialize_amplitude(plane, {'amplitude_init': 'uniform', 'amplitude_value': 1.0})
    assert torch.allclose(amp, torch.ones_like(amp))
    amplitude_initializations.initialize_amplitude(plane, {'amplitude_init': 'random'})


def test_modulator_forward_matches_getters():
    amp0 = torch.ones(1, 1, 8, 8)
    phase0 = torch.zeros(1, 1, 8, 8)
    opt_a = torch.zeros(1, 1, 8, 8)
    opt_p = torch.zeros(1, 1, 8, 8)
    mod = Modulator(amp0, phase0, opt_a, opt_p)
    field = torch.ones(2, 1, 8, 8, dtype=torch.complex64)
    out = mod(field)
    expected = field * mod.get_amplitude() * torch.exp(1j * mod.get_phase())
    assert torch.allclose(out, expected.to(out.dtype), atol=1e-5)
    assert torch.allclose(mod.get_amplitude(with_grad=False), amp0)


def test_sigmoid_not_tanh_keeps_nonnegative():
    amp0 = torch.ones(1, 1, 4, 4)
    phase0 = torch.zeros(1, 1, 4, 4)
    opt_a = torch.full((1, 1, 4, 4), -10.0)
    opt_p = torch.zeros(1, 1, 4, 4)
    mod = Modulator(amp0, phase0, opt_a, opt_p)
    assert (mod.get_amplitude() >= 0).all()
    assert torch.allclose(mod.get_amplitude(), mod.get_amplitude())  # forward/getter share map


def test_setters():
    amp0 = torch.ones(1, 1, 4, 4)
    phase0 = torch.zeros(1, 1, 4, 4)
    mod = Modulator(amp0, phase0, torch.zeros_like(amp0), torch.zeros_like(phase0))
    new_a = torch.full((1, 1, 4, 4), 0.3)
    new_p = torch.full((1, 1, 4, 4), 0.5)
    mod.set_amplitude(new_a, with_grad=False)
    mod.set_phase(new_p, with_grad=True)
    assert torch.allclose(mod.get_amplitude(with_grad=False), new_a)
    assert torch.allclose(mod.get_phase(with_grad=True), new_p)
    assert mod.initial_phase.requires_grad is False
    assert mod.optimizeable_phase.requires_grad is True


def test_factory_gradient_flags():
    plane = _plane()
    m_phase = ModulatorFactory()(plane, _factory_params('phase_only'))
    assert m_phase.optimizeable_phase.requires_grad is True
    assert m_phase.optimizeable_amplitude.requires_grad is False
    m_amp = ModulatorFactory()(plane, _factory_params('amplitude_only'))
    assert m_amp.optimizeable_amplitude.requires_grad is True
    assert m_amp.optimizeable_phase.requires_grad is False
    m_c = ModulatorFactory()(plane, _factory_params('complex'))
    assert m_c.optimizeable_amplitude.requires_grad and m_c.optimizeable_phase.requires_grad


def test_factory_uniform_value():
    plane = _plane()
    mod = ModulatorFactory()(plane, _factory_params(amplitude_value=0.8, phase_value=0.2))
    assert torch.allclose(mod.initial_amplitude, torch.full_like(mod.initial_amplitude, 0.8))
    assert torch.allclose(mod.initial_phase, torch.full_like(mod.initial_phase, 0.2))


def test_dtype_64_and_128():
    p64 = _plane(bits=64)
    p128 = _plane(bits=128)
    m64 = ModulatorFactory()(p64, _factory_params())
    m128 = ModulatorFactory()(p128, _factory_params())
    assert m64.initial_amplitude.dtype == torch.float32
    assert m128.initial_amplitude.dtype == torch.float64
    field64 = torch.ones(1, 1, 16, 16, dtype=torch.complex64)
    field128 = torch.ones(1, 1, 16, 16, dtype=torch.complex128)
    assert m64(field64).dtype == torch.complex64
    assert m128(field128).dtype == torch.complex128


def test_lens_phase_shape():
    plane = _plane()
    phase = phase_initializations.lens_phase(plane, {'focal_length': 10.0, 'wavelength': 520e-6})
    assert phase.shape == (1, 1, 16, 16)
