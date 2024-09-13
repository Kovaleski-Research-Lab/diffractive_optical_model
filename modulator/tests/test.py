import torch
import unittest
import sys
import yaml
sys.path.append('../')
sys.path.append('../../')

from initializations import phase_initializations, amplitude_initializations
from factory import ModulatorFactory
from modulator import Modulator
from plane import Plane

class TestModulator(unittest.TestCase):
    def setup(self):
        pass

    def tearDown(self):
        pass

    def test_uniform_phase_initialization(self):
        # Load the config
        config = yaml.safe_load(open('../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Initialize the phase
        phase = phase_initializations.initialize_phase({'phase_init': 'uniform'}, plane)

    def test_random_phase_initialization(self):
        # Load the config
        config = yaml.safe_load(open('../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Initialize the phase
        phase = phase_initializations.initialize_phase({'phase_init': 'random'}, plane)

    def test_uniform_amplitude_initialization(self):
        # Load the config
        config = yaml.safe_load(open('../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Initialize the amplitude
        amplitude = amplitude_initializations.initialize_amplitude({'amplitude_init': 'uniform'}, plane)

    def test_random_amplitude_initialization(self):
        # Load the config
        config = yaml.safe_load(open('../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Initialize the amplitude
        amplitude = amplitude_initializations.initialize_amplitude({'amplitude_init': 'random'}, plane)

    def test_modulator_init(self):
        amplitude = torch.ones(100,100)
        phase = torch.ones(100,100)
        modulator = Modulator(amplitude, phase)

    def test_modulator_set_amplitude_nograd(self):
        amplitude = torch.ones(100,100)
        phase = torch.ones(100,100)
        modulator = Modulator(amplitude, phase)
        new_amplitude = torch.zeros(100,100)
        modulator.set_amplitude(new_amplitude, with_grad=False)
        assert torch.allclose(modulator.amplitude, new_amplitude)
        assert modulator.amplitude.requires_grad == False

    def test_modulator_set_phase_nograd(self):
        amplitude = torch.ones(100,100)
        phase = torch.ones(100,100)
        modulator = Modulator(amplitude, phase)
        new_phase = torch.zeros(100,100)
        modulator.set_phase(new_phase, with_grad=False)
        assert torch.allclose(modulator.phase, new_phase)
        assert modulator.phase.requires_grad == False

    def test_modulator_set_amplitude_grad(self):
        amplitude = torch.ones(100,100)
        phase = torch.ones(100,100)
        modulator = Modulator(amplitude, phase)
        new_amplitude = torch.zeros(100,100)
        modulator.set_amplitude(new_amplitude, with_grad=True)
        assert torch.allclose(modulator.amplitude, new_amplitude)
        assert modulator.amplitude.requires_grad == True

    def test_modulator_set_phase_grad(self):
        amplitude = torch.ones(100,100)
        phase = torch.ones(100,100)
        modulator = Modulator(amplitude, phase)
        new_phase = torch.zeros(100,100)
        modulator.set_phase(new_phase, with_grad=True)
        assert torch.allclose(modulator.phase, new_phase)
        assert modulator.phase.requires_grad == True

    def test_modulator_get_phase_no_grad(self):
        amplitude = torch.rand(100,100, requires_grad=True)
        phase = torch.rand(100,100, requires_grad=True)
        modulator = Modulator(amplitude, phase)
        assert torch.allclose(modulator.get_phase(with_grad=False), phase)
        assert modulator.get_phase(with_grad=False).requires_grad == False

    def test_modulator_get_amplitude_no_grad(self):
        amplitude = torch.rand(100,100, requires_grad=True)
        phase = torch.rand(100,100, requires_grad=True)
        modulator = Modulator(amplitude, phase)
        assert torch.allclose(modulator.get_amplitude(with_grad=False), amplitude)
        assert modulator.get_amplitude(with_grad=False).requires_grad == False

    def test_modulator_get_phase_grad(self):
        amplitude = torch.rand(100,100, requires_grad=True)
        phase = torch.rand(100,100, requires_grad=True)
        modulator = Modulator(amplitude, phase)
        assert torch.allclose(modulator.get_phase(with_grad=True), phase)
        assert modulator.get_phase(with_grad=True).requires_grad == True

    def test_modulator_get_amplitude_grad(self):
        amplitude = torch.rand(100,100, requires_grad=True)
        phase = torch.rand(100,100, requires_grad=True)
        modulator = Modulator(amplitude, phase)
        assert torch.allclose(modulator.get_amplitude(with_grad=True), amplitude)
        assert modulator.get_amplitude(with_grad=True).requires_grad == True

    def test_modulator_forward(self):
        input_wavefront = torch.rand(100,100) * torch.exp(1j * torch.rand(100,100))
        amplitude = torch.rand(100,100)
        phase = torch.rand(100,100)
        modulator = Modulator(amplitude, phase)
        output_wavefront = modulator(input_wavefront)
        assert torch.allclose(output_wavefront, input_wavefront * amplitude * torch.exp(1j * phase))

    def test_factory_phase_only(self):
        # Load the config
        config = yaml.safe_load(open('../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Create some params for the factory
        params = {'gradients': 'phase_only', 'phase_init': 'random', 'amplitude_init': 'random'}
        # Call the factory
        modulator = ModulatorFactory()(params, plane)
        assert modulator.amplitude.requires_grad == False
        assert modulator.phase.requires_grad == True

    def test_factory_amplitude_only(self):
        # Load the config
        config = yaml.safe_load(open('../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Create some params for the factory
        params = {'gradients': 'amplitude_only', 'phase_init': 'random', 'amplitude_init': 'random'}
        # Call the factory
        modulator = ModulatorFactory()(params, plane)
        assert modulator.amplitude.requires_grad == True
        assert modulator.phase.requires_grad == False

    def test_factory_complex(self):
        # Load the config
        config = yaml.safe_load(open('../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Create some params for the factory
        params = {'gradients': 'complex', 'phase_init': 'random', 'amplitude_init': 'random'}
        # Call the factory
        modulator = ModulatorFactory()(params, plane)
        assert modulator.amplitude.requires_grad == True
        assert modulator.phase.requires_grad == True

def suite_basic():
    suite = unittest.TestSuite()
    suite.addTest(TestModulator('test_uniform_phase_initialization'))
    suite.addTest(TestModulator('test_random_phase_initialization'))
    suite.addTest(TestModulator('test_uniform_amplitude_initialization'))
    suite.addTest(TestModulator('test_random_amplitude_initialization'))
    suite.addTest(TestModulator('test_modulator_init'))
    suite.addTest(TestModulator('test_modulator_set_amplitude_nograd'))
    suite.addTest(TestModulator('test_modulator_set_phase_nograd'))
    suite.addTest(TestModulator('test_modulator_set_amplitude_grad'))
    suite.addTest(TestModulator('test_modulator_set_phase_grad'))
    suite.addTest(TestModulator('test_modulator_get_phase_no_grad'))
    suite.addTest(TestModulator('test_modulator_get_amplitude_no_grad'))
    suite.addTest(TestModulator('test_modulator_get_phase_grad'))
    suite.addTest(TestModulator('test_modulator_get_amplitude_grad'))
    suite.addTest(TestModulator('test_modulator_forward'))
    suite.addTest(TestModulator('test_factory_phase_only'))
    suite.addTest(TestModulator('test_factory_amplitude_only'))
    suite.addTest(TestModulator('test_factory_complex'))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite_basic())
