import torch
import torchvision
import pytorch_lightning as pl

class Propagator(pl.LightningModule):
    def __init__(self, input_plane, output_plane, fft_strategy, propagation_strategy):
        super().__init__()
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.fft_strategy = fft_strategy
        self.propagation_strategy = propagation_strategy
        self.register_buffer('H', propagation_strategy.get_transfer_function())
        self.cc_output = torchvision.transforms.CenterCrop((int(output_plane.Nx), int(output_plane.Ny)))
        self.cc_input = torchvision.transforms.CenterCrop((int(input_plane.Nx), int(input_plane.Ny)))
        padx = torch.div(input_plane.Nx, 2, rounding_mode='trunc')
        pady = torch.div(input_plane.Ny, 2, rounding_mode='trunc')
        self.padding = (pady, pady, padx, padx)

    def forward(self, input_wavefront):
        input_wavefront = torch.nn.functional.pad(input_wavefront, self.padding, mode="constant")
        return self.propagation_strategy.propagate(input_wavefront, self.fft_strategy)

