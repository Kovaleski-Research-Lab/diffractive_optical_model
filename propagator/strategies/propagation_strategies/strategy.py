from abc import ABC, abstractmethod

class PropagationStrategy(ABC):
    def __init__(self, input_plane, output_plane, wavelength):
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.wavelength = wavelength

    @abstractmethod
    def get_transfer_function(self):
        pass

    @abstractmethod
    def propagate(self, input_wavefront, fft_strategy):
        pass

