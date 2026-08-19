"""Sampling plane for scalar diffraction.

Coordinates use the FFT-standard grid (millimeters throughout):

    dx = Lx / Nx
    x  = (n - Nx // 2) * dx,   n = 0, ..., Nx - 1

so ``size`` is the computational window ``N * dx``. Padded axes double the
sample count at the same pitch. Tilted planes are not supported.
"""

import copy
import math

import torch
from loguru import logger
import numpy as np


_Z_HAT = (0.0, 0.0, 1.0)


class Plane:
    """A z-normal rectangular sampling plane.

    Parameters
    ----------
    params : dict
        ``name``, ``center`` (x, y, z), ``size`` (Lx, Ly), ``normal``,
        ``Nx``, ``Ny``. Units are millimeters.
    bits : int
        ``64``: float64 coordinates and complex64 fields.
        ``128``: float64 coordinates and complex128 fields.
        These are complex-bit widths, not IEEE-128 floats.
    """

    def __init__(self, params: dict, bits: int = 64) -> None:
        self._validate_params(params, bits)
        self.params = copy.deepcopy(params)
        self.name = params['name']
        logger.debug("Initializing plane {}".format(self.name))

        self.center_x, self.center_y, self.center_z = torch.tensor(params['center'])
        self.Lx, self.Ly = torch.tensor(params['size'])
        self.Nx = torch.tensor(params['Nx'])
        self.Ny = torch.tensor(params['Ny'])

        self.bits = bits
        self.fix_types(bits=self.bits)

        self.center = torch.tensor(
            [self.center_x, self.center_y, self.center_z],
            dtype=self.real_type_torch,
        )
        self.size = torch.tensor([self.Lx, self.Ly], dtype=self.real_type_torch)
        self.normal = torch.tensor(params['normal'], dtype=self.real_type_torch)
        self.normal = self.normal / torch.norm(self.normal)
        self._assert_z_normal()

        self.rot = torch.eye(3, dtype=self.real_type_torch)
        self.build_plane()

    @staticmethod
    def _validate_params(params, bits):
        if not isinstance(params, dict):
            raise TypeError("Plane params must be a dictionary.")
        required = {'name', 'center', 'size', 'normal', 'Nx', 'Ny'}
        missing = sorted(required.difference(params))
        if missing:
            raise ValueError("Missing required plane parameters: {}".format(", ".join(missing)))
        if bits not in (64, 128):
            raise ValueError(
                "Invalid bits={}. Use 64 (complex64 fields) or 128 (complex128 fields).".format(bits)
            )
        if not isinstance(params['name'], str) or not params['name'].strip():
            raise ValueError("Plane 'name' must be a non-empty string.")

        for key, length in (('center', 3), ('size', 2), ('normal', 3)):
            value = params[key]
            if not hasattr(value, '__len__') or len(value) != length:
                raise ValueError("Plane {!r} must contain {} values.".format(key, length))
            try:
                values = [float(v) for v in value]
            except (TypeError, ValueError) as exc:
                raise ValueError("Plane {!r} must contain numeric values.".format(key)) from exc
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Plane {!r} values must be finite.".format(key))

        if any(float(v) <= 0 for v in params['size']):
            raise ValueError("Plane sizes must be finite and strictly positive.")
        normal_norm = math.sqrt(sum(float(v) ** 2 for v in params['normal']))
        if normal_norm == 0:
            raise ValueError("Plane normal must be nonzero.")

        for key in ('Nx', 'Ny'):
            value = params[key]
            if isinstance(value, bool):
                raise ValueError("{} must be an integer of at least 2.".format(key))
            try:
                integer = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("{} must be an integer of at least 2.".format(key)) from exc
            if float(value) != integer or integer < 2:
                raise ValueError("{} must be an integer of at least 2.".format(key))

    def fix_types(self, bits=64):
        logger.debug("Fixing types for plane {}".format(self.name))
        # Coordinates are always float64. bits selects the field (complex) dtype.
        self.center_x = self.center_x.to(torch.float64)
        self.center_y = self.center_y.to(torch.float64)
        self.center_z = self.center_z.to(torch.float64)
        self.Lx = self.Lx.to(torch.float64)
        self.Ly = self.Ly.to(torch.float64)
        self.Nx = self.Nx.to(torch.int64)
        self.Ny = self.Ny.to(torch.int64)
        self.real_type_torch = torch.float64
        self.real_type_numpy = np.float64
        if bits == 128:
            self.complex_type_torch = torch.complex128
            self.complex_type_numpy = np.complex128
        elif bits == 64:
            self.complex_type_torch = torch.complex64
            self.complex_type_numpy = np.complex64
        else:
            logger.error("Invalid number of bits.")
            raise ValueError(
                "Invalid bits={}. Use 64 (complex64 fields) or 128 (complex128 fields).".format(bits)
            )

    def _assert_z_normal(self, atol=1e-6):
        z_hat = torch.tensor(_Z_HAT, dtype=self.normal.dtype, device=self.normal.device)
        if not torch.allclose(self.normal, z_hat, atol=atol):
            raise NotImplementedError(
                "Tilted planes are not supported. Plane {!r} has normal {}; "
                "only +z [0, 0, 1] is allowed.".format(self.name, self.normal.tolist())
            )

    def build_plane(self) -> None:
        logger.debug("Building plane {}".format(self.name))
        nx = int(self.Nx)
        ny = int(self.Ny)
        dtype = self.real_type_torch

        self.delta_x = self.Lx / self.Nx
        self.delta_y = self.Ly / self.Ny

        self.x = (torch.arange(nx, dtype=dtype) - nx // 2) * self.delta_x
        self.y = (torch.arange(ny, dtype=dtype) - ny // 2) * self.delta_y
        self.xx, self.yy = torch.meshgrid(self.x, self.y, indexing='ij')

        # Padded axes: 2N samples at the same pitch (window 2L).
        self.x_padded = (torch.arange(2 * nx, dtype=dtype) - nx) * self.delta_x
        self.y_padded = (torch.arange(2 * ny, dtype=dtype) - ny) * self.delta_y
        self.xx_padded, self.yy_padded = torch.meshgrid(
            self.x_padded, self.y_padded, indexing='ij'
        )

        self.fx = torch.fft.fftfreq(nx, d=float(self.delta_x), dtype=dtype)
        self.fy = torch.fft.fftfreq(ny, d=float(self.delta_y), dtype=dtype)
        self.fxx, self.fyy = torch.meshgrid(self.fx, self.fy, indexing='ij')
        self.delta_fx = self.fx[1] - self.fx[0]
        self.delta_fy = self.fy[1] - self.fy[0]

        self.fx_padded = torch.fft.fftfreq(2 * nx, d=float(self.delta_x), dtype=dtype)
        self.fy_padded = torch.fft.fftfreq(2 * ny, d=float(self.delta_y), dtype=dtype)
        self.fxx_padded, self.fyy_padded = torch.meshgrid(
            self.fx_padded, self.fy_padded, indexing='ij'
        )
        self.delta_fx_padded = self.fx_padded[1] - self.fx_padded[0]
        self.delta_fy_padded = self.fy_padded[1] - self.fy_padded[0]

    def _coordinate_tensors(self):
        names = (
            'center_x', 'center_y', 'center_z', 'Lx', 'Ly', 'Nx', 'Ny',
            'center', 'size', 'normal', 'rot',
            'delta_x', 'delta_y', 'x', 'y', 'xx', 'yy',
            'x_padded', 'y_padded', 'xx_padded', 'yy_padded',
            'fx', 'fy', 'fxx', 'fyy', 'delta_fx', 'delta_fy',
            'fx_padded', 'fy_padded', 'fxx_padded', 'fyy_padded',
            'delta_fx_padded', 'delta_fy_padded',
        )
        return names

    def to(self, device):
        """Move all coordinate tensors to ``device`` (for DNI on GPU)."""
        for name in self._coordinate_tensors():
            value = getattr(self, name, None)
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

    def print_info(self):
        logger.info("Plane {}:".format(self.name))
        logger.info("Center: {}".format(self.center))
        logger.info("Size (computational window N*dx): {}".format(self.size))
        logger.info("Samples: {}".format((self.Nx, self.Ny)))
        logger.info("delta_x, delta_y: {}, {}".format(self.delta_x, self.delta_y))
        logger.info("Normal vector: {}".format(self.normal))

    def is_same_spatial(self, plane):
        checks = [
            torch.isclose(self.Lx, plane.Lx),
            torch.isclose(self.Ly, plane.Ly),
            torch.isclose(self.Nx.float(), plane.Nx.float()),
            torch.isclose(self.Ny.float(), plane.Ny.float()),
            torch.isclose(self.delta_x, plane.delta_x),
            torch.isclose(self.delta_y, plane.delta_y),
        ]
        return all(bool(c) for c in checks)

    def is_smaller(self, plane):
        return bool(self.Lx < plane.Lx) and bool(self.Ly < plane.Ly)

    def scale(self, scale_factor, inplace=False):
        try:
            scale_factor = float(scale_factor)
        except (TypeError, ValueError) as exc:
            raise ValueError("scale_factor must be finite and strictly positive.") from exc
        if not math.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("scale_factor must be finite and strictly positive.")
        if inplace:
            self.Lx = self.Lx * scale_factor
            self.Ly = self.Ly * scale_factor
            self.size = torch.tensor([self.Lx, self.Ly], dtype=self.real_type_torch)
            self.params['size'] = [float(self.Lx), float(self.Ly)]
            self.build_plane()
            return self
        new_params = dict(self.params)
        new_params['size'] = [float(self.Lx * scale_factor), float(self.Ly * scale_factor)]
        return Plane(new_params, bits=self.bits)

    def __repr__(self):
        return "Plane: {}".format(self.name)
