import torch
from src.propagator.strategies.propagation_strategies.strategy import PropagationStrategy

class ASMStrategy(PropagationStrategy):
    def __init__(self, input_plane, output_plane, fft_strategy, wavelength):
        super().__init__(input_plane, output_plane, fft_strategy, wavelength)
        self.fft_strategy = fft_strategy
        self.transfer_function = self.get_transfer_function()

    def __repr__(self):
        return f"ASMStrategy(input_plane={self.input_plane}, output_plane={self.output_plane}, fft_strategy={self.fft_strategy}, wavelength={self.wavelength})"

    def get_transfer_function(self):
        fxx = self.fft_strategy.fxx
        fyy = self.fft_strategy.fyy

        mask = torch.sqrt(fxx**2 + fyy**2).real < (1 / self.wavelength)
        fxx = mask * fxx
        fyy = mask * fyy

        fz = torch.sqrt(1 - (self.wavelength * fxx)**2 - (self.wavelength * fyy)**2).real
        fz *= ((torch.pi * 2) / self.wavelength)

        distance = self.output_plane.center[-1] - self.input_plane.center[-1]

        H = torch.exp(1j * distance * fz)

        mag = H.abs()
        ang = H.angle()
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j * ang)

        shift = self.output_plane.center - self.input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]

        H = H * torch.exp(1j * 2 * torch.pi * (fxx * -x_shift + fyy * -y_shift))
        #H = torch.fft.fftshift(H)
        H = torch.reshape(H, (1, H.size(-2), H.size(-1)))
        H.requires_grad = False
        return H

    def propagate(self, input_wavefront):
        A = self.fft_strategy.fft2(input_wavefront)
        U = A * self.transfer_function
        U = self.fft_strategy.ifft2(U)
        return U

