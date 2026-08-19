"""Resample a field defined on plane0 onto plane1's pixel pitch and window."""

import torch
import torch.nn.functional as F


def center_crop(obj, crop_x, crop_y):
    obj_x, obj_y = obj.shape[-2:]
    start_x = obj_x // 2 - crop_x // 2
    start_y = obj_y // 2 - crop_y // 2
    return obj[:, :, start_x:start_x + crop_x, start_y:start_y + crop_y]


def crop_or_pad(obj, plane1):
    """Center-pad then center-crop so ``obj`` matches ``plane1`` sample counts."""
    target_x = int(plane1.Nx)
    target_y = int(plane1.Ny)
    _, _, obj_x, obj_y = obj.shape
    pad_x = max(target_x - obj_x, 0)
    pad_y = max(target_y - obj_y, 0)
    if pad_x or pad_y:
        # Preserve the sample at coordinate zero for both odd and even grids.
        top = target_x // 2 - obj_x // 2
        bottom = pad_x - top
        left = target_y // 2 - obj_y // 2
        right = pad_y - left
        obj = F.pad(obj, (left, right, top, bottom), mode='constant', value=0)
    return center_crop(obj, target_x, target_y)


def resample(plane0, obj, plane1):
    """Sample a real or complex field on ``plane1`` using global coordinates.

    Values outside ``plane0`` are zero. The first spatial tensor dimension is
    the plane x axis and the second is y, matching ``Plane``.
    """
    if not torch.is_tensor(obj) or obj.ndim != 4:
        raise ValueError("obj must have shape (batch, channel, Nx, Ny).")
    expected = (int(plane0.Nx), int(plane0.Ny))
    if tuple(obj.shape[-2:]) != expected:
        raise ValueError(
            "Object shape {} does not match source plane shape {}.".format(
                tuple(obj.shape[-2:]), expected
            )
        )
    if not (obj.is_floating_point() or obj.is_complex()):
        raise TypeError("obj must use a floating-point or complex dtype.")

    device = obj.device
    coordinate_dtype = torch.float64 if obj.dtype in (torch.float64, torch.complex128) else torch.float32
    x_in = plane0.x.to(device=device, dtype=coordinate_dtype)
    y_in = plane0.y.to(device=device, dtype=coordinate_dtype)
    shift = (plane1.center - plane0.center).to(device=device, dtype=coordinate_dtype)
    x_query = plane1.x.to(device=device, dtype=coordinate_dtype) + shift[0]
    y_query = plane1.y.to(device=device, dtype=coordinate_dtype) + shift[1]

    x_norm = 2 * (x_query - x_in[0]) / (x_in[-1] - x_in[0]) - 1
    y_norm = 2 * (y_query - y_in[0]) / (y_in[-1] - y_in[0]) - 1
    xx_norm, yy_norm = torch.meshgrid(x_norm, y_norm, indexing='ij')
    # grid_sample stores coordinates as (width, height) = (plane y, plane x).
    grid = torch.stack((yy_norm, xx_norm), dim=-1).unsqueeze(0)
    grid = grid.expand(obj.shape[0], -1, -1, -1)

    def sample(values):
        return F.grid_sample(
            values,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

    if obj.is_complex():
        return torch.complex(sample(obj.real), sample(obj.imag)).to(dtype=obj.dtype)
    return sample(obj)


def spatial_resample(plane0, obj, plane1):
    return resample(plane0, obj, plane1)


def create_cross_pattern(plane, cross_size):
    xx, yy = plane.xx, plane.yy
    cross = torch.zeros_like(xx)
    cross[(torch.abs(xx) < cross_size) | (torch.abs(yy) < cross_size)] = 1.0
    return cross.view(1, 1, int(plane.Nx), int(plane.Ny))
