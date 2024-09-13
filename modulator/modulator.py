import torch
import pytorch_lightning as pl

class Modulator(pl.LightningModule):
    def __init__(self, amplitude, phase):
        super().__init__()
        self.amplitude = torch.nn.Parameter(amplitude, amplitude.requires_grad)
        self.phase = torch.nn.Parameter(phase, phase.requires_grad)

    def forward(self, input_wavefront):
        return input_wavefront * self.amplitude * torch.exp(1j * self.phase)

    #-----------------#
    # Setters/Getters #
    #-----------------#
    def set_phase(self, phase, with_grad=True):
        if with_grad:
            self.phase = torch.nn.Parameter(phase, requires_grad=True)
        else:
            self.phase = torch.nn.Parameter(phase, requires_grad=False)

    def set_amplitude(self, amplitude, with_grad=True):
        if with_grad:
            self.amplitude = torch.nn.Parameter(amplitude, requires_grad=True)
        else:
            self.amplitude = torch.nn.Parameter(amplitude, requires_grad=False)

    def get_phase(self, with_grad=True):
        if with_grad:
            return self.phase
        else:
            return self.phase.detach()

    def get_amplitude(self, with_grad=True):
        if with_grad:
            return self.amplitude
        else:
            return self.amplitude.detach()




