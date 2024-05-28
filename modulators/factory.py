import yaml
from .modulator import Modulator
from .strategies.modulator_strategies.phase_only_strategy import PhaseOnlyStrategy
from .strategies.modulator_strategies.amplitude_only_strategy import AmplitudeOnlyStrategy
from .strategies.modulator_strategies.complex_strategy import ComplexStrategy

class ModulatorFactory:
    def __call__(self, params):
        return self.select_modulator(params)

    def select_modulator(self, params):
        mod_type = params.get('mod_type')

        if mod_type == 'phase_only':
            modulator_strategy = PhaseOnlyStrategy(params)
        elif mod_type == 'amplitude_only':
            modulator_strategy = AmplitudeOnlyStrategy(params)
        elif mod_type == 'complex':
            modulator_strategy = ComplexStrategy(params)
        else:
            raise ValueError(f"Invalid modulation type: {mod_type}")

        return Modulator(modulator_strategy)

