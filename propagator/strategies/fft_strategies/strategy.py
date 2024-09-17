from abc import ABC, abstractmethod

class FFTStrategy(ABC):
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

