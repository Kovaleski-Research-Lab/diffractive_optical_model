import numpy as np
from .strategy import FFTStrategy


class NumpyFFTStrategy(FFTStrategy):
    def fft(self, data):
        return np.fft.fft2(data)

    def ifft(self, data):
        return np.fft.ifft2(data)
