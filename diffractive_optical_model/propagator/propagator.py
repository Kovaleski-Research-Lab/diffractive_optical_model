import torch
import torch.nn as nn
import torch.nn.functional as F


def _crop_or_pad_at_origin(field, height, width, origin=None):
    """Resize the last two dimensions while preserving the coordinate-zero sample."""
    h, w = field.shape[-2:]
    if origin is None:
        origin = (h // 2, w // 2)
    origin_h, origin_w = origin
    target_origin_h = height // 2
    target_origin_w = width // 2

    pad_top = max(target_origin_h - origin_h, 0)
    pad_left = max(target_origin_w - origin_w, 0)
    pad_bottom = max(
        (height - target_origin_h - 1) - (h - origin_h - 1),
        0,
    )
    pad_right = max(
        (width - target_origin_w - 1) - (w - origin_w - 1),
        0,
    )
    if pad_top or pad_bottom or pad_left or pad_right:
        field = F.pad(field, (pad_left, pad_right, pad_top, pad_bottom))
        origin_h += pad_top
        origin_w += pad_left

    start_h = origin_h - target_origin_h
    start_w = origin_w - target_origin_w
    return field[..., start_h:start_h + height, start_w:start_w + width]


class Propagator(nn.Module):
    def __init__(self, input_plane, output_plane, fft_strategy, propagation_strategy, padded=False):
        super().__init__()
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.propagation_strategy = propagation_strategy
        self.padded = padded
        self.out_height = int(output_plane.Nx)
        self.out_width = int(output_plane.Ny)
        self.identity = (
            input_plane.is_same_spatial(output_plane)
            and bool(torch.allclose(input_plane.center, output_plane.center))
        )
        if padded:
            nx = int(input_plane.Nx)
            ny = int(input_plane.Ny)
            target_nx = int(fft_strategy.x_input.numel())
            target_ny = int(fft_strategy.y_input.numel())
            target_origin_x = int(torch.argmin(torch.abs(fft_strategy.x_input)).item())
            target_origin_y = int(torch.argmin(torch.abs(fft_strategy.y_input)).item())
            pad_x_before = target_origin_x - nx // 2
            pad_y_before = target_origin_y - ny // 2
            if pad_x_before < 0 or pad_y_before < 0:
                raise ValueError("The computational grid cannot be smaller than the input grid.")
            self.padding = (
                pad_y_before,
                target_ny - ny - pad_y_before,
                pad_x_before,
                target_nx - nx - pad_x_before,
            )
        else:
            self.padding = None
        x_output = fft_strategy.x_output
        y_output = fft_strategy.y_output
        self.output_origin = (
            int(torch.argmin(torch.abs(x_output)).item()),
            int(torch.argmin(torch.abs(y_output)).item()),
        )

    @property
    def fft_strategy(self):
        """The propagation strategy owns the shared FFT backend."""
        return self.propagation_strategy.fft_strategy

    def forward(self, input_wavefront):
        if not torch.is_tensor(input_wavefront):
            raise TypeError("input_wavefront must be a torch.Tensor.")
        if input_wavefront.ndim < 2:
            raise ValueError("input_wavefront must have at least two spatial dimensions.")
        expected = (int(self.input_plane.Nx), int(self.input_plane.Ny))
        if tuple(input_wavefront.shape[-2:]) != expected:
            raise ValueError(
                "Input field shape {} does not match plane {!r} spatial shape {}.".format(
                    tuple(input_wavefront.shape[-2:]),
                    self.input_plane.name,
                    expected,
                )
            )
        input_wavefront = input_wavefront.to(dtype=self.input_plane.complex_type_torch)
        if self.identity:
            return input_wavefront
        if self.padding is not None:
            input_wavefront = F.pad(input_wavefront, self.padding, mode="constant")
        return _crop_or_pad_at_origin(
            self.propagation_strategy.propagate(input_wavefront),
            self.out_height,
            self.out_width,
            self.output_origin,
        )
