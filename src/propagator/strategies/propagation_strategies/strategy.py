from abc import abstractmethod
import pytorch_lightning as pl

class PropagationStrategy(pl.LightningModule):
    def __init__(self, input_plane, output_plane, fft_strategy, wavelength):
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.fft_strategy = fft_strategy
        self.wavelength = wavelength

    @abstractmethod
    def get_transfer_function(self):
        pass

    @abstractmethod
    def propagate(self, input_wavefront):
        pass

