import yaml
from loguru import logger
from modulator.modulator import Modulator
from modulator.initializations.phase_initializations import initialize_phase
from modulator.initializations.amplitude_initializations import initialize_amplitude

class ModulatorFactory:
    def __call__(self, plane, params:dict={None:None}):
        self.gradients = params.get('gradients', 'none')
        amplitude, phase = self.initialize_amplitude_phase(plane, params)
        amplitude, phase = self.initialize_gradients(amplitude, phase)
        return Modulator(amplitude, phase)

    def initialize_amplitude_phase(self, plane, params):
        amplitude = initialize_amplitude(plane, params)
        phase = initialize_phase(plane, params)
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

