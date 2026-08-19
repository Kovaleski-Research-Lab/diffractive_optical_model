from abc import abstractmethod
import torch.nn as nn


class PropagationStrategy(nn.Module):
    def __init__(self, input_plane, output_plane, fft_strategy, wavelength):
        super().__init__()
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
