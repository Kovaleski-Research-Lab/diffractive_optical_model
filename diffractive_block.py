# Standard imports
import torch
from loguru import logger
import pytorch_lightning as pl

# Custom library imports
from plane import Plane
from modulator import ModulatorFactory
from propagator import PropagatorFactory

class DiffractiveBlock(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        input_plane_params = params['input_plane']
        output_plane_params = params['output_plane']
        modulator_params = params['modulator_params']
        propagator_params = params['propagator_params']

        input_plane = Plane(input_plane_params)
        output_plane = Plane(output_plane_params)

        self.modulator = ModulatorFactory(input_plane, modulator_params)
        self.propagator = PropagatorFactory(input_plane, output_plane, propagator_params)

    def forward(self, input_wavefront):
        return(self.propagator(self.modulator(input_wavefront)))




if __name__ == "__main__":

    input_plane_params = {'name':'plane0', 'center':[0,0,0], 'size':[8.96e-3, 8.96e-3], 'Nx':1080, 'Ny':1080}
    output_plane_params = {'name':'plane1', 'center':[0,0,3.e-2], 'size':[8.96e-3, 8.96e-3], 'Nx':1080, 'Ny':1080}



    db = DiffractiveBlock(params)
