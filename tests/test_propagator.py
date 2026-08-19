import torch

from diffractive_optical_model.plane.plane import Plane
from diffractive_optical_model.propagator.factory import PropagatorFactory
from diffractive_optical_model.propagator.propagator import Propagator


def test_propagator_does_not_duplicate_H():
    plane = Plane({
        'name': 'p',
        'center': [0, 0, 0],
        'size': [1.0, 1.0],
        'normal': [0, 0, 1],
        'Nx': 16,
        'Ny': 16,
    })
    out = Plane({
        'name': 'o',
        'center': [0, 0, 0.02],
        'size': [1.0, 1.0],
        'normal': [0, 0, 1],
        'Nx': 16,
        'Ny': 16,
    })
    prop = PropagatorFactory()(plane, out, {
        'prop_type': 'asm',
        'fft_type': 'pytorch',
        'wavelength': 0.00052,
        'padded': True,
    })
    assert not hasattr(prop, 'H') or 'H' not in dict(prop.named_buffers())
    assert 'transfer_function' in dict(prop.propagation_strategy.named_buffers())
    assert isinstance(prop, Propagator)
    assert prop.padded is True
