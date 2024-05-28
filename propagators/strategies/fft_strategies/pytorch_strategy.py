import torch
from .strategy import FFTStrategy

class PyTorchFFTStrategy(FFTStrategy):
    def fft(self, data):
        return torch.fft.fft(data)

    def ifft(self, data):
        return torch.fft.ifft(data)

