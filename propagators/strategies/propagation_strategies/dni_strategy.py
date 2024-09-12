import torch
from .strategy import PropagationStrategy

class DNIStrategy(PropagationStrategy):
    def get_transfer_function(self):
        pass

    def propagate(self, input_wavefront, fft_strategy):
        z_distance = self.input_plane.center[-1] - self.output_plane.center[-1]
        k = torch.pi * 2 / self.wavelength
        output_field = input_wavefront.new_empty(input_wavefront.size(), dtype=torch.complex64)

        for i, x in enumerate(self.input_plane.x_padded):
            for j, y in enumerate(self.input_plane.y_padded):
                r = torch.sqrt((self.output_plane.xx_padded - x)**2 + (self.output_plane.yy_padded - y)**2 + z_distance**2)
                chirp = torch.exp(1j * k * r)
                scalar1 = z_distance / r
                scalar2 = ((1 / r) - 1j * k)
                combined = input_wavefront * chirp * scalar1 * scalar2
                output_field[:, :, i, j] = combined.sum()

        return output_field

