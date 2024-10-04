import numpy as np
from strategy import FFTStrategy


class NumpyFFTStrategy(FFTStrategy):
    def fft(self, data):
        return np.fft.fft(data)

    def ifft(self, data):
        return np.fft.ifft(data)

    def fft2(self, data):
        return np.fft.fft2(data)

    def ifft2(self, data):
        return np.fft.ifft2(data)

