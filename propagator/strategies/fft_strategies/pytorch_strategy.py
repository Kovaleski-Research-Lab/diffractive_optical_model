import torch
from strategy import FFTStrategy

class PyTorchFFTStrategy(FFTStrategy):
    def fft(self, data):
        return torch.fft.fft(data)

    def ifft(self, data):
        return torch.fft.ifft(data)

    def fft2(self, data):
        return torch.fft.fft2(data)

    def ifft2(self, data):
        return torch.fft.ifft2(data)
