import torch
from loguru import logger
from diffractive_optical_model.propagator.strategies.propagation_strategies.strategy import PropagationStrategy
from diffractive_optical_model.propagator.strategies.propagation_strategies.rsc_strategy import (
    rayleigh_sommerfeld_kernel,
)


class DNIStrategy(PropagationStrategy):
    """Direct numerical integration of the RS-I kernel. Reference solver only."""

    def __init__(self, input_plane, output_plane, fft_strategy, wavelength, chunk=64):
        super().__init__(input_plane, output_plane, fft_strategy, wavelength)
        self.chunk = chunk
        # Dummy buffer so the module has a transfer_function attribute.
        self.register_buffer('transfer_function', torch.ones(1, 1, 1))

    def get_transfer_function(self):
        return self.transfer_function

    def propagate(self, input_wavefront):
        z = self.output_plane.center[-1] - self.input_plane.center[-1]
        xx_in = self.fft_strategy.xx_input
        yy_in = self.fft_strategy.yy_input
        xx_out = self.fft_strategy.xx_output
        yy_out = self.fft_strategy.yy_output
        dx = self.input_plane.delta_x
        dy = self.input_plane.delta_y
        shift = self.output_plane.center - self.input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]

        device = input_wavefront.device
        xx_in = xx_in.to(device=device, dtype=torch.float64)
        yy_in = yy_in.to(device=device, dtype=torch.float64)
        xx_out = xx_out.to(device=device)
        yy_out = yy_out.to(device=device)
        dx = dx.to(device=device)
        dy = dy.to(device=device)
        z = z.to(device=device)
        x_shift = x_shift.to(device=device)
        y_shift = y_shift.to(device=device)

        x_out = xx_out.reshape(-1)
        y_out = yy_out.reshape(-1)
        n_out = x_out.numel()
        interactions = n_out * xx_in.numel()
        if interactions > 100_000_000:
            logger.warning(
                "DNI will evaluate {:,} source-output interactions; "
                "this reference calculation may be very slow.".format(interactions)
            )

        leading = input_wavefront.shape[:-2]
        if input_wavefront.is_complex():
            complex_dtype = input_wavefront.dtype
        else:
            complex_dtype = self.input_plane.complex_type_torch
        source = input_wavefront.to(dtype=complex_dtype)
        output = torch.empty(
            *leading,
            xx_out.size(0),
            xx_out.size(1),
            dtype=complex_dtype,
            device=device,
        )
        output_flat = output.reshape(*leading, n_out)

        for start in range(0, n_out, self.chunk):
            end = min(start + self.chunk, n_out)
            xo = x_out[start:end].to(dtype=torch.float64)[:, None, None]
            yo = y_out[start:end].to(dtype=torch.float64)[:, None, None]
            h = rayleigh_sommerfeld_kernel(
                xo - xx_in + x_shift, yo - yy_in + y_shift, z, self.wavelength, dx, dy
            ).to(dtype=complex_dtype)
            # h: (P, Hin, Win); source: (..., Hin, Win)
            contrib = (source.unsqueeze(-3) * h).sum(dim=(-2, -1))
            output_flat[..., start:end] = contrib

        return output
