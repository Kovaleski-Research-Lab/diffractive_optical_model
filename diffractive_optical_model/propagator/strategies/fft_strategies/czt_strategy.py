"""Torch-native scaled Fourier sums on arbitrary equispaced spatial grids.

The forward transform is deliberately unnormalised, like ``torch.fft.fft``::

    G(f_k) = sum_n g(x_n) exp(-2 pi i f_k x_n).

The inverse uses the Fourier-integral quadrature ``dx_input * df``.  This is
``1 / N`` when the spatial and frequency grids are a reciprocal DFT pair, so
same-grid transforms have the usual exact FFT round trip.

Each one-dimensional sum is evaluated with Bluestein's identity and one linear
convolution.  No dense DFT matrices are formed.
"""

import math

import torch
import torch.nn as nn

from diffractive_optical_model.propagator.strategies.fft_strategies.strategy import (
    FFTStrategy,
)


def _complex_dtype(dtype):
    if dtype == torch.complex128 or dtype == torch.float64:
        return torch.complex128
    if dtype.is_complex or dtype.is_floating_point:
        return torch.complex64
    raise TypeError("CZT transforms require floating-point or complex input")


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def _integer_ceiling(value):
    """Ceil a positive float without adding a bin for round-off at integers."""
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=1e-12, abs_tol=1e-12):
        return max(1, int(nearest))
    return max(1, int(math.ceil(value)))


class _BluesteinFourierSum(nn.Module):
    """Evaluate ``sum_n z[n] exp(sign*2*pi*i*a[n]*b[k])`` along one axis."""

    def __init__(
        self,
        sample_origin,
        sample_step,
        sample_count,
        evaluation_origin,
        evaluation_step,
        evaluation_count,
        sign,
        scale=1.0,
    ):
        super().__init__()
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 or +1")

        self.sample_count = int(sample_count)
        self.evaluation_count = int(evaluation_count)
        self.fft_length = _next_power_of_two(
            self.sample_count + self.evaluation_count - 1
        )

        # Build in double precision. At execution these O(N+M) buffers are cast
        # to the input complex dtype, preserving complex64 and complex128.
        dtype = torch.float64
        a0 = torch.as_tensor(sample_origin, dtype=dtype)
        da = torch.as_tensor(sample_step, dtype=dtype)
        b0 = torch.as_tensor(evaluation_origin, dtype=dtype)
        db = torch.as_tensor(evaluation_step, dtype=dtype)
        n = torch.arange(self.sample_count, dtype=dtype)
        k = torch.arange(self.evaluation_count, dtype=dtype)
        signed_pi = sign * torch.pi
        cross_step = da * db

        pre_angle = signed_pi * (2 * da * b0 * n + cross_step * n.square())
        post_angle = signed_pi * (
            2 * a0 * (b0 + db * k) + cross_step * k.square()
        )

        offsets = torch.arange(
            -(self.sample_count - 1), self.evaluation_count, dtype=dtype
        )
        kernel_angle = -signed_pi * cross_step * offsets.square()

        pre = torch.polar(torch.ones_like(pre_angle), pre_angle)
        post = torch.polar(torch.ones_like(post_angle), post_angle)
        post = post * torch.as_tensor(scale, dtype=dtype)

        kernel = torch.zeros(self.fft_length, dtype=torch.complex128)
        kernel_indices = offsets.to(torch.int64).remainder(self.fft_length)
        kernel[kernel_indices] = torch.polar(
            torch.ones_like(kernel_angle), kernel_angle
        )
        kernel_fft = torch.fft.fft(kernel)

        self.register_buffer("pre_chirp", pre, persistent=False)
        self.register_buffer("kernel_fft", kernel_fft, persistent=False)
        self.register_buffer("post_chirp", post, persistent=False)

    def forward(self, values, axis=-1):
        axis = axis % values.ndim
        if values.shape[axis] != self.sample_count:
            raise ValueError(
                "Expected transform axis of length "
                f"{self.sample_count}, got {values.shape[axis]}"
            )

        dtype = _complex_dtype(values.dtype)
        values = values.to(dtype=dtype).movedim(axis, -1)
        pre = self.pre_chirp.to(device=values.device, dtype=dtype)
        kernel_fft = self.kernel_fft.to(device=values.device, dtype=dtype)
        post = self.post_chirp.to(device=values.device, dtype=dtype)

        work = values * pre
        convolution = torch.fft.ifft(
            torch.fft.fft(work, n=self.fft_length, dim=-1) * kernel_fft,
            dim=-1,
        )
        result = convolution[..., : self.evaluation_count] * post
        return result.movedim(-1, axis)


class CZTStrategy(FFTStrategy):
    """Separable Bluestein/CZT backend for equispaced input/output planes.

    Frequencies are a common, monotonically increasing, centered grid. Its
    periodic bandwidth is the Nyquist bandwidth of the coarser spatial pitch,
    while its spacing is no larger than the reciprocal of the larger physical
    support. ``fft``/``ifft`` use the x axis; ``fft2``/``ifft2`` transform x
    and y on the final two tensor dimensions.
    """

    def __init__(self, input_plane, output_plane, kwargs=None):
        super().__init__()
        if kwargs is None:
            kwargs = {}
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.padded = kwargs.get("padded", False)

        self._pick_x_y()
        self._pick_fx_fy()
        self._create_transforms()

    def __repr__(self):
        return (
            f"CZTStrategy(input_plane={self.input_plane}, "
            f"output_plane={self.output_plane}, padded={self.padded})"
        )

    def _set_coordinate(self, name, value):
        # Geometry is derived state and should move with the module without
        # bloating checkpoints.
        self.register_buffer(name, value, persistent=False)

    def _pick_x_y(self):
        suffix = "_padded" if self.padded else ""
        x_input = getattr(self.input_plane, "x" + suffix)
        y_input = getattr(self.input_plane, "y" + suffix)
        x_output = getattr(self.output_plane, "x" + suffix)
        y_output = getattr(self.output_plane, "y" + suffix)

        self._set_coordinate("x_input", x_input)
        self._set_coordinate("y_input", y_input)
        self._set_coordinate("x_output", x_output)
        self._set_coordinate("y_output", y_output)

        xx_input, yy_input = torch.meshgrid(x_input, y_input, indexing="ij")
        xx_output, yy_output = torch.meshgrid(x_output, y_output, indexing="ij")
        self._set_coordinate("xx_input", xx_input)
        self._set_coordinate("yy_input", yy_input)
        self._set_coordinate("xx_output", xx_output)
        self._set_coordinate("yy_output", yy_output)

    @staticmethod
    def _frequency_axis(
        input_axis, output_axis, input_support, output_support
    ):
        dx_input = float(input_axis[1] - input_axis[0])
        dx_output = float(output_axis[1] - output_axis[0])
        coarse_pitch = max(abs(dx_input), abs(dx_output))
        support = max(float(input_support), float(output_support))

        # K * df is the represented periodic bandwidth. This construction
        # simultaneously gives K*df = 1/dx_coarse and df <= 1/support.
        count = _integer_ceiling(support / coarse_pitch)
        # ``min`` also makes both constraints true under floating-point
        # round-off when support/coarse_pitch is mathematically integral.
        spacing = min(1.0 / (count * coarse_pitch), 1.0 / support)
        frequency = (
            torch.arange(count, dtype=input_axis.dtype, device=input_axis.device)
            - count // 2
        ) * spacing
        return frequency, spacing

    def _pick_fx_fy(self):
        padding_factor = 2 if self.padded else 1
        fx, delta_fx = self._frequency_axis(
            self.x_input,
            self.x_output,
            padding_factor * self.input_plane.Lx,
            padding_factor * self.output_plane.Lx,
        )
        fy, delta_fy = self._frequency_axis(
            self.y_input,
            self.y_output,
            padding_factor * self.input_plane.Ly,
            padding_factor * self.output_plane.Ly,
        )
        self._set_coordinate("fx", fx)
        self._set_coordinate("fy", fy)
        self._set_coordinate(
            "delta_fx", torch.as_tensor(delta_fx, dtype=fx.dtype, device=fx.device)
        )
        self._set_coordinate(
            "delta_fy", torch.as_tensor(delta_fy, dtype=fy.dtype, device=fy.device)
        )
        fxx, fyy = torch.meshgrid(fx, fy, indexing="ij")
        self._set_coordinate("fxx", fxx)
        self._set_coordinate("fyy", fyy)

    @staticmethod
    def _axis_parameters(axis):
        return axis[0], axis[1] - axis[0], axis.numel()

    def _create_axis_transforms(
        self, input_axis, output_axis, frequency_axis, delta_frequency
    ):
        input_origin, input_step, input_count = self._axis_parameters(input_axis)
        output_origin, output_step, output_count = self._axis_parameters(output_axis)
        frequency_origin, frequency_step, frequency_count = self._axis_parameters(
            frequency_axis
        )

        forward = _BluesteinFourierSum(
            input_origin,
            input_step,
            input_count,
            frequency_origin,
            frequency_step,
            frequency_count,
            sign=-1,
        )
        inverse = _BluesteinFourierSum(
            frequency_origin,
            frequency_step,
            frequency_count,
            output_origin,
            output_step,
            output_count,
            sign=1,
            scale=input_step.abs() * delta_frequency,
        )
        return forward, inverse

    def _create_transforms(self):
        self.forward_x, self.inverse_x = self._create_axis_transforms(
            self.x_input, self.x_output, self.fx, self.delta_fx
        )
        self.forward_y, self.inverse_y = self._create_axis_transforms(
            self.y_input, self.y_output, self.fy, self.delta_fy
        )

    def fft(self, data):
        return self.forward_x(data, axis=-1)

    def ifft(self, data):
        return self.inverse_x(data, axis=-1)

    def fft2(self, data):
        transformed = self.forward_y(data, axis=-1)
        return self.forward_x(transformed, axis=-2)

    def ifft2(self, data):
        transformed = self.inverse_y(data, axis=-1)
        return self.inverse_x(transformed, axis=-2)


# Descriptive alias for callers that use the existing ``*FFTStrategy`` naming.
CZTFFTStrategy = CZTStrategy
