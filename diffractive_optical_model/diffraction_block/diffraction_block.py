import torch
import torch.nn as nn

from diffractive_optical_model.plane.plane import Plane
from diffractive_optical_model.modulator.factory import ModulatorFactory
from diffractive_optical_model.propagator.factory import PropagatorFactory


class DiffractionBlock(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.input_plane_params = params['input_plane']
        self.output_plane_params = params['output_plane']
        self.modulator_params = params['modulator_params']
        self.propagator_params = params['propagator_params']
        self.bits = params.get('bits', self.propagator_params.get('bits', 64))

        self.input_plane = Plane(self.input_plane_params, bits=self.bits)
        self.output_plane = Plane(self.output_plane_params, bits=self.bits)
        self.modulator = ModulatorFactory()(self.input_plane, self.modulator_params)
        self.propagator = PropagatorFactory()(self.input_plane, self.output_plane, self.propagator_params)

    def forward(self, input_wavefront):
        input_wavefront = input_wavefront.to(self.input_plane.complex_type_torch)
        modulated_wavefront = self.modulator(input_wavefront)
        return self.propagator(modulated_wavefront)
