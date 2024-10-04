import torch
import unittest
import sys
import yaml
sys.path.append('../')
sys.path.append('../../../')

from diffractive_optical_model.modulator.initializations import phase_initializations, amplitude_initializations
from diffractive_optical_model.modulator.factory import ModulatorFactory
from diffractive_optical_model.modulator.modulator import Modulator
from diffractive_optical_model.plane.plane import Plane

class TestModulator(unittest.TestCase):
    def setup(self):
        pass

    def tearDown(self):
        pass

    def test_uniform_phase_initialization(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Initialize the phase
        phase = phase_initializations.initialize_phase(plane, {'phase_init': 'uniform'})

    def test_random_phase_initialization(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Initialize the phase
        phase = phase_initializations.initialize_phase(plane, {'phase_init': 'random'})

    def test_uniform_amplitude_initialization(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Initialize the amplitude
        amplitude = amplitude_initializations.initialize_amplitude(plane, {'amplitude_init': 'uniform'})

    def test_random_amplitude_initialization(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Initialize the amplitude
        amplitude = amplitude_initializations.initialize_amplitude(plane, {'amplitude_init': 'random'})

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
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Create some params for the factory
        params = {'gradients': 'phase_only', 
                  'amplitude_init': 'random',
                  'amplitude_value': 1.0,
                  'phase_init': 'random', 
                  'phase_value': 0.0,
                  'focal_length': 'none',
                  'wavelength': 520.e-6}

        # Call the factory
        modulator = ModulatorFactory()(plane, params)
        assert modulator.amplitude.requires_grad == False
        assert modulator.phase.requires_grad == True

    def test_factory_amplitude_only(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Create some params for the factory
        params = {'gradients': 'amplitude_only', 
                  'amplitude_init': 'random',
                  'amplitude_value': 1.0,
                  'phase_init': 'random', 
                  'phase_value': 0.0,
                  'focal_length': 'none',
                  'wavelength': 520.e-6}
        # Call the factory
        modulator = ModulatorFactory()(plane, params)
        assert modulator.amplitude.requires_grad == True
        assert modulator.phase.requires_grad == False

    def test_factory_complex(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Create some params for the factory
        params = {'gradients': 'complex', 
                  'amplitude_init': 'random',
                  'amplitude_value': 1.0,
                  'phase_init': 'random', 
                  'phase_value': 0.0,
                  'focal_length': 'none',
                  'wavelength': 520.e-6}
        # Call the factory
        modulator = ModulatorFactory()(plane, params)
        assert modulator.amplitude.requires_grad == True
        assert modulator.phase.requires_grad == True


    def test_init_dtype_64(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config, bits=64)
        # Create some params for the factory
        params = {'gradients': 'complex', 
                  'amplitude_init': 'random',
                  'amplitude_value': 1.0,
                  'phase_init': 'random', 
                  'phase_value': 0.0,
                  'focal_length': 'none',
                  'wavelength': 520.e-6}
        # Call the factory
        modulator = ModulatorFactory()(plane, params)
        assert modulator.amplitude.dtype == torch.float32
        assert modulator.phase.dtype == torch.float32

    def test_init_dtype_128(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config, bits=128)
        # Create some params for the factory
        params = {'gradients': 'complex', 
                  'amplitude_init': 'random',
                  'amplitude_value': 1.0,
                  'phase_init': 'random', 
                  'phase_value': 0.0,
                  'focal_length': 'none',
                  'wavelength': 520.e-6}
        # Call the factory
        modulator = ModulatorFactory()(plane, params)
        assert modulator.amplitude.dtype == torch.float64
        assert modulator.phase.dtype == torch.float64

    def test_factory_modulator_forward_64bit(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config, bits=64)
        # Create some params for the factory
        params = {'gradients': 'complex', 
                  'amplitude_init': 'random',
                  'amplitude_value': 1.0,
                  'phase_init': 'random', 
                  'phase_value': 0.0,
                  'focal_length': 'none',
                  'wavelength': 520.e-6}
        # Call the factory
        modulator = ModulatorFactory()(plane, params)
        # Create some input wavefront
        amplitude = torch.rand(plane.Nx, plane.Ny, dtype=torch.float32)
        phase = torch.rand(plane.Nx, plane.Ny, dtype=torch.float32)
        input_wavefront = amplitude * torch.exp(1j * phase)
        shape = input_wavefront.shape
        input_wavefront = input_wavefront.view(1,1,shape[0],shape[1]).to(torch.complex64)
        output_wavefront = modulator(input_wavefront)

        test_modulator = modulator.amplitude * torch.exp(1j * modulator.phase)
        assert torch.allclose(output_wavefront, test_modulator * input_wavefront)
        assert output_wavefront.shape == input_wavefront.shape
        assert output_wavefront.dtype == torch.complex64

    def test_factory_modulator_forward_128bit(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config, bits=128)
        # Create some params for the factory
        params = {'gradients': 'complex', 
                  'amplitude_init': 'random',
                  'amplitude_value': 1.0,
                  'phase_init': 'random', 
                  'phase_value': 0.0,
                  'focal_length': 'none',
                  'wavelength': 520.e-6}
        # Call the factory
        modulator = ModulatorFactory()(plane, params)
        # Create some input wavefront
        amplitude = torch.rand(plane.Nx, plane.Ny, dtype=torch.float64)
        phase = torch.rand(plane.Nx, plane.Ny, dtype=torch.float64)
        input_wavefront = amplitude * torch.exp(1j * phase)
        shape = input_wavefront.shape
        input_wavefront = input_wavefront.view(1,1,shape[0],shape[1]).to(torch.complex128)
        output_wavefront = modulator(input_wavefront)

        test_modulator = modulator.amplitude * torch.exp(1j * modulator.phase)
        assert torch.allclose(output_wavefront, test_modulator * input_wavefront)
        assert output_wavefront.shape == input_wavefront.shape
        assert output_wavefront.dtype == torch.complex128

    def test_factory_uniform_init_withValue(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Create some params for the factory
        value = 3.0
        params = {'gradients': 'complex', 
                  'amplitude_init': 'uniform',
                  'amplitude_value': value,
                  'phase_init': 'uniform', 
                  'phase_value': value,
                  'focal_length': 'none',
                  'wavelength': 520.e-6}
        # Call the factory
        modulator = ModulatorFactory()(plane, params)
        assert torch.allclose(modulator.amplitude, torch.ones(plane.Nx, plane.Ny)*value)
        assert torch.allclose(modulator.phase, torch.ones(plane.Nx, plane.Ny)*value)

    def test_cuda(self):
        # Load the config
        config = yaml.safe_load(open('../../../config.yaml', 'r'))
        # Get a plane config
        plane_config = config['planes'][0]
        # Initialize a plane
        plane = Plane(plane_config)
        # Create some params for the factory
        value = 3.0
        params = {'gradients': 'complex', 
                  'amplitude_init': 'uniform',
                  'amplitude_value': value,
                  'phase_init': 'uniform', 
                  'phase_value': value,
                  'focal_length': 'none',
                  'wavelength': 520.e-6}
        # Call the factory
        modulator = ModulatorFactory()(plane, params)
        modulator = modulator.to('cuda')
        assert modulator.amplitude.is_cuda
        assert modulator.phase.is_cuda

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
    suite.addTest(TestModulator('test_init_dtype_64'))
    suite.addTest(TestModulator('test_init_dtype_128'))
    suite.addTest(TestModulator('test_factory_modulator_forward_64bit'))
    suite.addTest(TestModulator('test_factory_modulator_forward_128bit'))
    suite.addTest(TestModulator('test_factory_uniform_init_withValue'))
    suite.addTest(TestModulator('test_cuda'))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite_basic())
