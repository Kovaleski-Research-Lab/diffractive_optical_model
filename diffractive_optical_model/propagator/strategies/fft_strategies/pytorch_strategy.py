import torch
from diffractive_optical_model.propagator.strategies.fft_strategies.strategy import FFTStrategy


class PyTorchFFTStrategy(FFTStrategy):
    def __init__(self, input_plane, output_plane, kwargs=None):
        super().__init__()
        if kwargs is None:
            kwargs = {}
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.padded = kwargs.get('padded', False)
        self.pad_total_x = int(kwargs.get('_pad_total_x', int(input_plane.Nx)))
        self.pad_total_y = int(kwargs.get('_pad_total_y', int(input_plane.Ny)))
        self.pick_fx_fy()
        self.pick_x_y()

    def __repr__(self):
        return (
            f"PyTorchFFTStrategy(input_plane={self.input_plane}, "
            f"output_plane={self.output_plane}, padded={self.padded})"
        )

    def pick_fx_fy(self):
        if self.padded:
            nx = int(self.input_plane.Nx) + self.pad_total_x
            ny = int(self.input_plane.Ny) + self.pad_total_y
            dtype = self.input_plane.real_type_torch
            self.fx = torch.fft.fftfreq(nx, d=float(self.input_plane.delta_x), dtype=dtype)
            self.fy = torch.fft.fftfreq(ny, d=float(self.input_plane.delta_y), dtype=dtype)
            self.fxx, self.fyy = torch.meshgrid(self.fx, self.fy, indexing='ij')
            self.delta_fx = self.fx[1] - self.fx[0]
            self.delta_fy = self.fy[1] - self.fy[0]
        else:
            self.fx = self.input_plane.fx
            self.fxx = self.input_plane.fxx
            self.fy = self.input_plane.fy
            self.fyy = self.input_plane.fyy
            self.delta_fx = self.input_plane.delta_fx
            self.delta_fy = self.input_plane.delta_fy

    def pick_x_y(self):
        if self.padded:
            nx = int(self.input_plane.Nx) + self.pad_total_x
            ny = int(self.input_plane.Ny) + self.pad_total_y
            dtype = self.input_plane.real_type_torch
            self.x_input = (
                torch.arange(nx, dtype=dtype) - nx // 2
            ) * self.input_plane.delta_x
            self.y_input = (
                torch.arange(ny, dtype=dtype) - ny // 2
            ) * self.input_plane.delta_y
            self.xx_input, self.yy_input = torch.meshgrid(
                self.x_input, self.y_input, indexing='ij'
            )
            # This backend is restricted to identical spatial grids.
            self.x_output = self.x_input
            self.y_output = self.y_input
            self.xx_output = self.xx_input
            self.yy_output = self.yy_input
        else:
            self.x_input = self.input_plane.x
            self.xx_input = self.input_plane.xx
            self.y_input = self.input_plane.y
            self.yy_input = self.input_plane.yy
            self.x_output = self.output_plane.x
            self.xx_output = self.output_plane.xx
            self.y_output = self.output_plane.y
            self.yy_output = self.output_plane.yy

    def fft(self, data):
        data = torch.fft.ifftshift(data, dim=-1)
        return torch.fft.fft(data, dim=-1)

    def ifft(self, data):
        data = torch.fft.ifft(data, dim=-1)
        return torch.fft.fftshift(data, dim=-1)

    def fft2(self, data):
        data = torch.fft.ifftshift(data, dim=(-2, -1))
        return torch.fft.fft2(data)

    def ifft2(self, data):
        data = torch.fft.ifft2(data)
        return torch.fft.fftshift(data, dim=(-2, -1))
