from abc import ABC, abstractmethod

class FFTStrategy(ABC):
    @abstractmethod
    def fft(self, data):
        pass

    @abstractmethod
    def ifft(self, data):
        pass

