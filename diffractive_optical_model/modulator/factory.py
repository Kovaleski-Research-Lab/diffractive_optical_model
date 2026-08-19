import torch
from loguru import logger
from diffractive_optical_model.modulator.modulator import Modulator
from diffractive_optical_model.modulator.initializations.phase_initializations import initialize_phase
from diffractive_optical_model.modulator.initializations.amplitude_initializations import initialize_amplitude


class ModulatorFactory:
    def __call__(self, plane, params=None):
        if params is None:
            params = {}
        self.gradients = str(params.get('gradients', 'none')).lower()
        allowed_gradients = ('none', 'phase_only', 'amplitude_only', 'complex')
        if self.gradients not in allowed_gradients:
            raise ValueError(
                "Unsupported gradients mode {!r}; use one of {}.".format(
                    self.gradients, ", ".join(allowed_gradients)
                )
            )
        initial_amplitude, initial_phase = self.initialize_amplitude_phase(plane, params)
        initial_amplitude.requires_grad_(False)
        initial_phase.requires_grad_(False)

        amplitude = torch.zeros_like(initial_amplitude)
        phase = torch.zeros_like(initial_phase)
        amplitude, phase = self.initialize_gradients(amplitude, phase)
        return Modulator(initial_amplitude, initial_phase, amplitude, phase)

    def initialize_amplitude_phase(self, plane, params):
        return initialize_amplitude(plane, params), initialize_phase(plane, params)

    def initialize_gradients(self, amplitude, phase):
        if self.gradients == 'phase_only':
            logger.info("Phase only optimization")
            phase.requires_grad = True
            amplitude.requires_grad = False
        elif self.gradients == 'amplitude_only':
            logger.info("Amplitude only optimization")
            phase.requires_grad = False
            amplitude.requires_grad = True
        elif self.gradients == 'complex':
            logger.info("Phase and amplitude optimization")
            phase.requires_grad = True
            amplitude.requires_grad = True
        elif self.gradients == 'none':
            logger.info("No modulator optimization")
            phase.requires_grad = False
            amplitude.requires_grad = False
        else:
            raise ValueError(
                "Unsupported gradients mode {!r}; use none, phase_only, "
                "amplitude_only, or complex.".format(self.gradients)
            )
        return amplitude, phase
