import numpy as np
import pytest
import torch

from diffractive_optical_model.plane.plane import Plane


def _params(name='plane0', size=(8.64e3, 8.64e3), Nx=32, Ny=32, center=(0, 0, 0), normal=(0, 0, 1)):
    return {
        'name': name,
        'center': list(center),
        'size': list(size),
        'normal': list(normal),
        'Nx': Nx,
        'Ny': Ny,
    }


def test_dx_equals_L_over_N():
    plane = Plane(_params(size=(8.0, 4.0), Nx=16, Ny=8))
    assert torch.isclose(plane.delta_x, plane.Lx / plane.Nx)
    assert torch.isclose(plane.delta_y, plane.Ly / plane.Ny)
    assert torch.isclose(plane.Nx * plane.delta_x, plane.Lx)
    assert torch.isclose(plane.Ny * plane.delta_y, plane.Ly)


def test_centered_fft_grid():
    plane = Plane(_params(Nx=16, Ny=16, size=(8.0, 8.0)))
    nx = int(plane.Nx)
    assert torch.isclose(plane.x[nx // 2], torch.tensor(0.0, dtype=plane.x.dtype))
    assert torch.isclose(plane.x[0], -plane.Lx / 2)
    assert torch.isclose(plane.x[-1], plane.Lx / 2 - plane.delta_x)


def test_padded_delta_matches_unpadded():
    plane = Plane(_params(Nx=16, Ny=8, size=(8.0, 4.0)))
    assert plane.x_padded.numel() == 2 * int(plane.Nx)
    assert plane.y_padded.numel() == 2 * int(plane.Ny)
    padded_dx = plane.x_padded[1] - plane.x_padded[0]
    padded_dy = plane.y_padded[1] - plane.y_padded[0]
    assert torch.isclose(padded_dx, plane.delta_x)
    assert torch.isclose(padded_dy, plane.delta_y)
    assert torch.isclose(plane.delta_fx_padded, plane.fx_padded[1] - plane.fx_padded[0])


def test_fx_matches_fftfreq():
    plane = Plane(_params(Nx=16, Ny=8, size=(8.0, 4.0)))
    fx = torch.fft.fftfreq(int(plane.Nx), d=float(plane.delta_x), dtype=plane.real_type_torch)
    fy = torch.fft.fftfreq(int(plane.Ny), d=float(plane.delta_y), dtype=plane.real_type_torch)
    assert torch.allclose(plane.fx, fx)
    assert torch.allclose(plane.fy, fy)
    fx_p = torch.fft.fftfreq(2 * int(plane.Nx), d=float(plane.delta_x), dtype=plane.real_type_torch)
    assert torch.allclose(plane.fx_padded, fx_p)


def test_plane_precisions_coords_match():
    plane0 = Plane(_params(), bits=64)
    plane1 = Plane(_params(), bits=128)
    assert plane0.complex_type_torch == torch.complex64
    assert plane1.complex_type_torch == torch.complex128
    assert torch.allclose(plane0.x, plane1.x)
    assert torch.allclose(plane0.fx, plane1.fx)
    assert torch.allclose(plane0.x_padded, plane1.x_padded)


def test_is_same_spatial():
    plane0 = Plane(_params(name='a', Nx=32, Ny=32))
    plane1 = Plane(_params(name='b', Nx=32, Ny=32))
    plane2 = Plane(_params(name='c', Nx=16, Ny=32))
    plane3 = Plane(_params(name='d', Nx=32, Ny=16))
    assert plane0.is_same_spatial(plane1)
    assert not plane0.is_same_spatial(plane2)
    assert not plane0.is_same_spatial(plane3)


def test_is_smaller():
    large = Plane(_params(size=(8.0, 8.0)))
    small = Plane(_params(name='small', size=(4.0, 4.0)))
    assert not large.is_smaller(small)
    assert small.is_smaller(large)


def test_scale_notinplace():
    plane0 = Plane(_params(size=(8.0, 8.0), Nx=16, Ny=16))
    plane1 = Plane(_params(size=(8.0, 8.0), Nx=16, Ny=16))
    scale = 0.6
    plane2 = plane0.scale(scale, inplace=False)
    assert plane0.is_same_spatial(plane1)
    assert torch.isclose(plane0.Lx * scale, plane2.Lx)
    assert torch.isclose(plane0.Ly * scale, plane2.Ly)
    assert plane0.Nx == plane2.Nx
    assert torch.isclose(plane2.Nx * plane2.delta_x, plane2.Lx)
    assert torch.isclose(plane2.x[0], -plane2.Lx / 2)
    assert torch.isclose(plane2.x[-1], plane2.Lx / 2 - plane2.delta_x)
    assert plane2.params['size'][0] == pytest.approx(float(plane2.Lx))


def test_scale_inplace():
    plane0 = Plane(_params(size=(8.0, 8.0), Nx=16, Ny=16))
    scale = 0.6
    plane2 = plane0.scale(scale, inplace=False)
    plane0.scale(scale, inplace=True)
    assert plane0.is_same_spatial(plane2)
    assert torch.isclose(plane0.size[0], plane0.Lx)


def test_tilted_plane_rejected():
    with pytest.raises(NotImplementedError):
        Plane(_params(normal=(0, 1, 0)))
    with pytest.raises(NotImplementedError):
        Plane(_params(normal=(0, 0, -1)))


def test_invalid_bits():
    with pytest.raises(ValueError):
        Plane(_params(), bits=32)


def test_to_device_cpu():
    plane = Plane(_params())
    plane.to('cpu')
    assert plane.x.device.type == 'cpu'
    assert plane.fx_padded.device.type == 'cpu'


@pytest.mark.parametrize(
    'updates,match',
    [
        ({'Nx': 1}, 'at least 2'),
        ({'Ny': 0}, 'at least 2'),
        ({'size': [0.0, 1.0]}, 'strictly positive'),
        ({'size': [float('nan'), 1.0]}, 'finite'),
        ({'normal': [0.0, 0.0, 0.0]}, 'nonzero'),
        ({'center': [0.0, float('inf'), 0.0]}, 'finite'),
    ],
)
def test_invalid_geometry_rejected_early(updates, match):
    params = _params()
    params.update(updates)
    with pytest.raises(ValueError, match=match):
        Plane(params)


def test_missing_plane_parameter_has_actionable_error():
    params = _params()
    del params['normal']
    with pytest.raises(ValueError, match='normal'):
        Plane(params)


def test_scale_rejects_nonpositive_or_nonfinite_values():
    plane = Plane(_params())
    for value in (0, -1, float('nan')):
        with pytest.raises(ValueError, match='strictly positive'):
            plane.scale(value)
