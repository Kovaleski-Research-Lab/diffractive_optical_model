import torch
import pytorch_lightning as pl

class Modulator(pl.LightningModule):
    def __init__(self, amplitude, phase):
        super().__init__()
        self.amplitude = torch.nn.Parameter(amplitude, amplitude.requires_grad)
        self.phase = torch.nn.Parameter(phase, phase.requires_grad)

    def forward(self, input_wavefront):
        return input_wavefront * self.amplitude * torch.exp(1j * self.phase)


