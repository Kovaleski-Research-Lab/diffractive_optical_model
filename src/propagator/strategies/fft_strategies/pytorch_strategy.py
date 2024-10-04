import torch
from propagator.strategies.fft_strategies.strategy import FFTStrategy

class PyTorchFFTStrategy(FFTStrategy):
    def __init__(self, input_plane, output_plane, kwargs:dict={None:None}):
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.padded = kwargs.get('padded', False)
        self.pick_fx_fy()
        self.pick_x_y()

    def __repr__(self):
        return f"PyTorchFFTStrategy(input_plane={self.input_plane}, output_plane={self.output_plane})"

    def pick_fx_fy(self):
        if self.padded:
            self.fx = self.input_plane.fx_padded
            self.fxx = self.input_plane.fxx_padded
            self.fy = self.input_plane.fy_padded
            self.fyy = self.input_plane.fyy_padded
        else:
            self.fx = self.input_plane.fx
            self.fxx = self.input_plane.fxx
            self.fy = self.input_plane.fy
            self.fyy = self.input_plane.fyy

    def pick_x_y(self):
        if self.padded:
            self.x_input = self.input_plane.x_padded
            self.xx_input = self.input_plane.xx_padded
            self.y_input = self.input_plane.y_padded
            self.yy_input = self.input_plane.yy_padded

            self.x_output = self.output_plane.x_padded
            self.xx_output = self.output_plane.xx_padded
            self.y_output = self.output_plane.y_padded
            self.yy_output = self.output_plane.yy_padded
        else:
            self.x_input = self.input_plane.x
            self.xx_input = self.input_plane.xx
            self.y_input = self.input_plane.y
            self.yy_input = self.input_plane

            self.x_output = self.output_plane.x
            self.xx_output = self.output_plane.xx
            self.y_output = self.output_plane.y
            self.yy_output = self.output_plane.yy

    def fft(self, data):
        return torch.fft.fft(data)

    def ifft(self, data):
        return torch.fft.ifft(data)

    def fft2(self, data):
        return torch.fft.fftn(data)

    def ifft2(self, data):
        return torch.fft.ifftn(data)
