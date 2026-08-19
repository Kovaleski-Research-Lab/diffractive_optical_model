from abc import abstractmethod
import torch.nn as nn


class FFTStrategy(nn.Module):
    """FFT backend. Spatial arrays are centered (origin at N//2).

    ``fft2`` / ``ifft2`` own the origin convention:
    ifftshift -> transform last two dims -> fftshift on the inverse.
    Frequencies from ``fftfreq`` (origin at index 0) are used for multiply.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fft(self, data):
        pass

    @abstractmethod
    def ifft(self, data):
        pass

    @abstractmethod
    def fft2(self, data):
        pass

    @abstractmethod
    def ifft2(self, data):
        pass
