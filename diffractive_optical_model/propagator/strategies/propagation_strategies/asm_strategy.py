import torch
from diffractive_optical_model.propagator.strategies.propagation_strategies.strategy import PropagationStrategy


class ASMStrategy(PropagationStrategy):
    """Angular spectrum method with a complex kz (evanescent waves decay).

    Time convention is exp(-j ω t), matching H = exp(j kz z) for propagating
    components. No magnitude normalization. An optional Matsushima band-limit
    zeros samples of H whose chirp aliases on the frequency grid.
    """

    def __init__(self, input_plane, output_plane, fft_strategy, wavelength, band_limit=True):
        super().__init__(input_plane, output_plane, fft_strategy, wavelength)
        self.band_limit = band_limit
        H = self.get_transfer_function().detach()
        self.register_buffer('transfer_function', H)

    def __repr__(self):
        return (
            f"ASMStrategy(input_plane={self.input_plane}, output_plane={self.output_plane}, "
            f"fft_strategy={self.fft_strategy}, wavelength={self.wavelength})"
        )

    def get_transfer_function(self):
        fxx = self.fft_strategy.fxx
        fyy = self.fft_strategy.fyy
        wavelength = self.wavelength
        if not torch.is_tensor(wavelength):
            wavelength = torch.tensor(wavelength, dtype=fxx.dtype)

        arg = 1 - (wavelength * fxx) ** 2 - (wavelength * fyy) ** 2
        # Principal sqrt: Im(kz) >= 0 so evanescent waves decay for z > 0.
        kz = (2 * torch.pi / wavelength) * torch.sqrt(arg.to(torch.complex128))
        kz = kz.to(torch.complex128)

        z = self.output_plane.center[-1] - self.input_plane.center[-1]
        z = z.to(dtype=torch.float64)
        # Decay in |z| for both forward and backward hops.
        H = torch.exp(1j * kz.real * z - kz.imag.abs() * z.abs())

        shift = self.output_plane.center - self.input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]
        # Local output coordinate x_o represents global x_o + center_out.
        # Therefore the kernel separation contains +(center_out-center_in).
        H = H * torch.exp(1j * 2 * torch.pi * (fxx * x_shift + fyy * y_shift))

        if self.band_limit:
            H = H * self._matsushima_band_limit(
                fxx, fyy, wavelength, z, x_shift, y_shift
            )

        H = H.reshape(1, H.size(-2), H.size(-1))
        return H.to(self.input_plane.complex_type_torch)

    def _matsushima_band_limit(self, fxx, fyy, wavelength, z, x_shift=0.0, y_shift=0.0):
        """Zero H where axial chirp plus the geometric shift ramp aliases."""
        delta_fx = self.fft_strategy.delta_fx.abs()
        delta_fy = self.fft_strategy.delta_fy.abs()
        fr2 = fxx ** 2 + fyy ** 2
        propagating = fr2 < (1 / wavelength) ** 2
        kz_norm = torch.sqrt(torch.clamp(1 / wavelength ** 2 - fr2, min=0.0))
        kz_norm = torch.clamp(kz_norm, min=1e-30)
        displacement_x = x_shift - z * fxx / kz_norm
        displacement_y = y_shift - z * fyy / kz_norm
        limit_x = displacement_x.abs() <= (1.0 / (2.0 * delta_fx))
        limit_y = displacement_y.abs() <= (1.0 / (2.0 * delta_fy))
        alias_ok = limit_x & limit_y
        # Evanescent bins keep the complex-kz decay; only propagating chirp is band-limited.
        keep = torch.where(propagating, alias_ok, torch.ones_like(alias_ok))
        return keep.to(dtype=fxx.dtype)

    def propagate(self, input_wavefront):
        A = self.fft_strategy.fft2(input_wavefront)
        U = A * self.transfer_function
        return self.fft_strategy.ifft2(U)
