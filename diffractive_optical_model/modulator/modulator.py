import torch
import pytorch_lightning as pl

class Modulator(pl.LightningModule):
    def __init__(self, initial_amplitude, initial_phase, optimizeable_amplitude, optimizeable_phase):
        super().__init__()
        self.initial_amplitude = torch.nn.Parameter(initial_amplitude, requires_grad=False)
        self.initial_phase = torch.nn.Parameter(initial_phase, requires_grad=False)
        self.optimizeable_amplitude = torch.nn.Parameter(optimizeable_amplitude, requires_grad=optimizeable_amplitude.requires_grad)
        self.optimizeable_phase = torch.nn.Parameter(optimizeable_phase, requires_grad=optimizeable_phase.requires_grad)

    def forward(self, input_wavefront):
        # Combine the initial and optimizeable parameters into the final parameters
        amplitude = self.initial_amplitude + torch.nn.functional.sigmoid(self.optimizeable_amplitude)
        phase = self.initial_phase + (torch.nn.functional.sigmoid(self.optimizeable_phase) * 2 * torch.pi)
        modulator = amplitude * torch.exp(1j * phase)
        return input_wavefront * modulator

    #-----------------#
    # Setters/Getters #
    #-----------------#
    def set_phase(self, phase, with_grad=True):
        if with_grad:
            self.initial_phase = torch.nn.Parameter(phase, requires_grad=True)
        else:
            self.initial_phase = torch.nn.Parameter(phase, requires_grad=False)

    def set_amplitude(self, amplitude, with_grad=True):
        if with_grad:
            self.initial_amplitude = torch.nn.Parameter(amplitude, requires_grad=True)
        else:
            self.initial_amplitude = torch.nn.Parameter(amplitude, requires_grad=False)

    def get_phase(self, with_grad=True):
        phase = self.initial_phase + (torch.nn.functional.sigmoid(self.optimizeable_phase) * 2 * torch.pi)
        if with_grad:
            return phase
        else:
            return phase.detach()

    def get_amplitude(self, with_grad=True):
        amplitude = self.initial_amplitude + torch.nn.functional.sigmoid(self.optimizeable_amplitude)
        if with_grad:
            return amplitude
        else:
            return amplitude.detach()




