import yaml
import torch
from loguru import logger
from .propagator import Propagator
from .strategies.fft_strategies.numpy_strategy import NumpyFFTStrategy
from .strategies.fft_strategies.pytorch_strategy import PyTorchFFTStrategy
from .strategies.fft_strategies.mtp_strategy import MTPFFTStrategy
from .strategies.propagation_strategies.asm_strategy import ASMStrategy
from .strategies.propagation_strategies.rsc_strategy import RSCStrategy

class PropagatorFactory:
    def __call__(self, input_plane, output_plane, params):
        return self.select_propagator(input_plane, output_plane, params)

    def select_propagator(self, input_plane, output_plane, params):
        # FFT Strategy selection
        if params['fft_type'] == 'numpy':
            fft_strategy = NumpyFFTStrategy()
        elif params['fft_type'] == 'pytorch':
            fft_strategy = PyTorchFFTStrategy()
        elif params['fft_type'] == 'mtp':
            fft_strategy = MTPFFTStrategy(input_plane, output_plane)
        else:
            raise ValueError(f"Invalid FFT type: {params['fft_type']}")

        # Propagation Strategy selection
        prop_type = params['prop_type']
        if prop_type is None:
            if self.check_asm_distance(input_plane, output_plane, params):
                propagation_strategy = ASMStrategy(input_plane, output_plane, params['wavelength'])
            else:
                propagation_strategy = RSCStrategy(input_plane, output_plane, params['wavelength'])
        else:
            if prop_type == 'asm':
                propagation_strategy = ASMStrategy(input_plane, output_plane, params['wavelength'])
            elif prop_type == 'rsc':
                propagation_strategy = RSCStrategy(input_plane, output_plane, params['wavelength'])
            else:
                raise ValueError(f"Invalid propagation type: {prop_type}")

        return Propagator(input_plane, output_plane, fft_strategy, propagation_strategy)

    def check_asm_distance(self, input_plane, output_plane, params):
        logger.debug("Checking ASM propagation criteria")
        wavelength = torch.tensor(params['wavelength'])
        delta_x = input_plane.delta_x
        delta_y = input_plane.delta_y
        Nx = input_plane.Nx
        Ny = input_plane.Ny

        shift_x = output_plane.center[0] - input_plane.center[0]
        shift_y = output_plane.center[1] - input_plane.center[1]
        distance = output_plane.center[-1] - input_plane.center[-1]
        distance = torch.abs(distance)

        logger.debug(f"Axial distance between input and output planes: {distance}")
        logger.debug(f"Shift between input and output planes: [{shift_x}, {shift_y}]")

        distance_criteria_y = 2 * delta_y * (Ny * delta_y - shift_y) / wavelength
        distance_criteria_y *= torch.sqrt(1 - (wavelength / (2 * Ny))**2)
        distance_criteria_y = torch.abs(distance_criteria_y)

        distance_criteria_x = 2 * delta_x * (Nx * delta_x - shift_x) / wavelength
        distance_criteria_x *= torch.sqrt(1 - (wavelength / (2 * Nx))**2)
        distance_criteria_x = torch.abs(distance_criteria_x)

        strict_distance = torch.min(distance_criteria_y, distance_criteria_x)
        logger.debug(f"Maximum axial distance for ASM: {strict_distance}")

        return torch.le(distance, strict_distance)

