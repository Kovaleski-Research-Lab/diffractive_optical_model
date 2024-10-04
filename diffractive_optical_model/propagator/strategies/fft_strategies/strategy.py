from abc import abstractmethod
import pytorch_lightning as pl

class FFTStrategy(pl.LightningModule):
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

