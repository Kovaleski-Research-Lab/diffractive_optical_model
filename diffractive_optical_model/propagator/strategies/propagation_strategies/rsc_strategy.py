import torch
from diffractive_optical_model.propagator.strategies.propagation_strategies.strategy import PropagationStrategy


def rayleigh_sommerfeld_kernel(xx, yy, z, wavelength, dx, dy, eps=1e-30):
    """Bilateral outgoing RS-I kernel for the exp(-i omega t) convention.

    For z != 0, ``h(-z) = conj(h(+z))``. The z=0 distributional limit is
    handled as identity by ``Propagator`` rather than evaluated pointwise.
    """
    if not torch.is_tensor(wavelength):
        wavelength = torch.tensor(wavelength, dtype=xx.dtype, device=xx.device)
    else:
        wavelength = wavelength.to(dtype=xx.dtype, device=xx.device)
    z = torch.as_tensor(z, dtype=xx.dtype, device=xx.device)
    r = torch.sqrt(xx ** 2 + yy ** 2 + z ** 2)
    r = torch.clamp(r, min=eps)
    k = 2 * torch.pi / wavelength
    sign_z = torch.sign(z)
    h = torch.exp(sign_z * 1j * k * r) / r
    h = h * ((1 / r) - (sign_z * 1j * k))
    h = h * (torch.abs(z) / r)
    h = h * (1 / (2 * torch.pi))
    h = h * dx * dy
    return h


class RSCStrategy(PropagationStrategy):
    """Rayleigh-Sommerfeld convolution (Goodman; JOSAA 401908 eq. 29)."""

    def __init__(self, input_plane, output_plane, fft_strategy, wavelength):
        super().__init__(input_plane, output_plane, fft_strategy, wavelength)
        H = self.get_transfer_function().detach()
        self.register_buffer('transfer_function', H)

    def __repr__(self):
        return (
            f"RSCStrategy(input_plane={self.input_plane}, output_plane={self.output_plane}, "
            f"fft_strategy={self.fft_strategy}, wavelength={self.wavelength})"
        )

    def get_transfer_function(self):
        xx = self.fft_strategy.xx_input
        yy = self.fft_strategy.yy_input
        z = self.output_plane.center[-1] - self.input_plane.center[-1]
        shift = self.output_plane.center - self.input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]
        dx = self.input_plane.delta_x
        dy = self.input_plane.delta_y
        h_rsc = rayleigh_sommerfeld_kernel(
            xx + x_shift, yy + y_shift, z, self.wavelength, dx, dy
        )
        H = self.fft_strategy.fft2(h_rsc)
        H = torch.reshape(H, (1, H.size(-2), H.size(-1)))
        return H.to(self.input_plane.complex_type_torch)

    def propagate(self, input_wavefront):
        A = self.fft_strategy.fft2(input_wavefront)
        U = A * self.transfer_function
        return self.fft_strategy.ifft2(U)
