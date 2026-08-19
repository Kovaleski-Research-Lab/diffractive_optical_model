import pytest
import torch

from diffractive_optical_model.plane.plane import Plane
from diffractive_optical_model.utils.spatial_resample import spatial_resample, crop_or_pad


def _plane(name, size, Nx, Ny, center=(0, 0, 0)):
    return Plane({
        'name': name,
        'center': list(center),
        'size': list(size),
        'normal': [0, 0, 1],
        'Nx': Nx,
        'Ny': Ny,
    })


def test_resample_center_stays_center():
    p0 = _plane('a', (2.0, 2.0), 32, 32)
    p1 = _plane('b', (1.0, 1.0), 32, 32)
    obj = torch.zeros(1, 1, 32, 32)
    obj[..., 16, 16] = 1.0
    out = spatial_resample(p0, obj, p1)
    assert out.shape == (1, 1, 32, 32)
    peak = out.squeeze()
    ix = int(torch.argmax(peak.reshape(-1)).item())
    i, j = divmod(ix, 32)
    assert abs(i - 16) <= 2
    assert abs(j - 16) <= 2


def test_odd_size_center_crop():
    obj = torch.ones(1, 1, 5, 5)
    plane = _plane('s', (1.0, 1.0), 3, 3)
    cropped = crop_or_pad(obj, plane)
    assert cropped.shape == (1, 1, 3, 3)
    # Center of 5x5 is index 2; crop of 3 takes indices 1:4.
    assert torch.allclose(cropped, torch.ones(1, 1, 3, 3))


def test_odd_size_center_pad():
    obj = torch.ones(1, 1, 2, 2)
    plane = _plane('s', (1.0, 1.0), 5, 5)
    padded = crop_or_pad(obj, plane)
    assert padded.shape == (1, 1, 5, 5)
    # 3 pixels of pad in each dim: 1 on one side, 2 on the other (//2).
    assert padded[0, 0, 1:3, 1:3].sum() == 4 or padded[0, 0, 2:4, 2:4].sum() == 4


def test_physical_peak_location():
    # Feature at x=0 (grid center) on a larger plane remains at x=0 on a finer/smaller sensor.
    p0 = _plane('obj', (4.0, 4.0), 40, 40)
    p1 = _plane('sen', (2.0, 2.0), 20, 20)
    obj = torch.zeros(1, 1, 40, 40)
    obj[..., 20, 20] = 1.0
    out = spatial_resample(p0, obj, p1)
    peak = out.squeeze()
    ix = int(torch.argmax(peak).item() // 20)
    iy = int(torch.argmax(peak).item() % 20)
    x_peak = float(p1.x[ix])
    y_peak = float(p1.y[iy])
    assert abs(x_peak) <= 2 * float(p1.delta_x)
    assert abs(y_peak) <= 2 * float(p1.delta_y)


def test_complex_resample_preserves_phase_and_precision():
    p0 = _plane('obj', (2.0, 2.0), 32, 32)
    p1 = _plane('sen', (1.0, 1.0), 16, 16)
    field = torch.full((2, 1, 32, 32), 1j, dtype=torch.complex128)
    out = spatial_resample(p0, field, p1)
    assert out.dtype == torch.complex128
    assert out.shape == (2, 1, 16, 16)
    assert torch.allclose(out, torch.full_like(out, 1j), atol=1e-12)


def test_resample_uses_global_plane_centers():
    p0 = _plane('obj', (1.0, 1.0), 16, 16)
    p1 = _plane('sen', (1.0, 1.0), 16, 16, center=(0.25, 0, 0))
    field = torch.zeros(1, 1, 16, 16)
    field[..., 8, 8] = 1.0
    out = spatial_resample(p0, field, p1)
    peak_index = torch.unravel_index(out.abs().argmax(), out.shape)
    local_x = float(p1.x[peak_index[-2]])
    global_x = local_x + float(p1.center_x)
    assert abs(global_x) <= float(p1.delta_x)


def test_resample_rejects_wrong_shape():
    p0 = _plane('obj', (1.0, 1.0), 16, 16)
    p1 = _plane('sen', (1.0, 1.0), 8, 8)
    with pytest.raises(ValueError, match='does not match'):
        spatial_resample(p0, torch.ones(1, 1, 8, 8), p1)
