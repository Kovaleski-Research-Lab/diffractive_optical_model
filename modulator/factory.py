import yaml
from loguru import logger
from modulator import Modulator
from initializations.phase_initializations import initialize_phase
from initializations.amplitude_initializations import initialize_amplitude

class ModulatorFactory:
    def __call__(self, params, plane):
        self.gradients = params['gradients']
        amplitude, phase = self.initialize_amplitude_phase(params, plane)
        amplitude, phase = self.initialize_gradients(amplitude, phase)
        return Modulator(amplitude, phase)

    def initialize_amplitude_phase(self, params, plane):
        amplitude = initialize_amplitude(params, plane)
        phase = initialize_phase(params, plane)
        return amplitude, phase

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
            logger.warning("modulator_type not specified. Setting no modulator optimization")
            phase.requires_grad = False
            amplitude.requires_grad = False
            #Log a non-critical error here

        return amplitude, phase

