import torch
from .strategy import PropagationStrategy

class RSCStrategy(PropagationStrategy):
    def __init__(self, input_plane, output_plane, wavelength):
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.wavelength = wavelength
        self.H = self.get_transfer_function()
    def get_transfer_function(self):
        xx, yy = self.input_plane.xx_padded, self.input_plane.yy_padded

        fxx, fyy = self.input_plane.fxx_padded, self.input_plane.fyy_padded

        mask = torch.sqrt(fxx**2 + fyy**2).real < (1 / self.wavelength)
        fxx = mask * fxx
        fyy = mask * fyy

        distance = self.output_plane.center[-1] - self.input_plane.center[-1]
        shift = self.output_plane.center - self.input_plane.center
        x_shift = shift[0]
        y_shift = shift[1]

        r = torch.sqrt((xx - x_shift)**2 + (yy - y_shift)**2 + distance**2).real
        k = (2 * torch.pi / self.wavelength)
        z = distance

        h_rsc = torch.exp(torch.sign(distance) * 1j * k * r) / r
        h_rsc *= ((1 / r) - (1j * k))
        h_rsc *= (1 / (2 * torch.pi)) * (z / r)

        H = torch.fft.fftn(h_rsc)

        mag = H.abs()
        ang = H.angle()
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j * ang)
        H = torch.fft.fftshift(H)

        H.requires_grad = False
        return H

    def propagate(self, input_wavefront, fft_strategy):
        A = fft_strategy.fft(input_wavefront)
        A = torch.fft.fftshift(A)
        U = A * self.H
        U = fft_strategy.ifft(U)
        U = torch.fft.ifftshift(U, dim=(-1, -2))
        return U

