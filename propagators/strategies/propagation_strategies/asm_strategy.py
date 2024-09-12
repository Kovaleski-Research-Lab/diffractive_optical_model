import torch
from .strategy import PropagationStrategy

class ASMStrategy(PropagationStrategy):
    def get_transfer_function(self):
        input_dx = self.input_plane.delta_x.real
        input_dy = self.input_plane.delta_y.real
        output_dx = self.output_plane.delta_x.real
        output_dy = self.output_plane.delta_y.real

        if input_dx > output_dx:
            fxx = self.input_plane.fxx_padded
        else:
            fxx = self.output_plane.fxx_padded

        if input_dy > output_dy:
            fyy = self.input_plane.fyy_padded
        else:
            fyy = self.output_plane.fyy_padded

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
        H = torch.fft.fftshift(H)
        H.requires_grad = False

        return H

    def propagate(self, input_wavefront, fft_strategy):
        A = fft_strategy.fft(input_wavefront)
        A = torch.fft.fftshift(A)
        U = A * self.get_transfer_function()
        U = torch.fft.ifftshift(U, dim=(-1, -2))
        U = fft_strategy.ifft(U)
        return U

