# Standard imports
import torch
from loguru import logger
import pytorch_lightning as pl

# Custom library imports
from plane import Plane
from modulator import lensPhase, ModulatorFactory
from propagator import PropagatorFactory

class DiffractiveBlock(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        input_plane_params = params['input_plane']
        output_plane_params = params['output_plane']
        modulator_params = params['modulator_params']

        input_plane = Plane(input_plane_params)
        output_plane = Plane(output_plane_params)

        modulator_factory = ModulatorFactory()
        propagator_factory = PropagatorFactory()

        self.modulator = modulator_factory(input_plane, modulator_params)
        self.propagator = propagator_factory(input_plane, output_plane)

    def forward(self, input_wavefront):
        return(self.propagator(self.modulator(input_wavefront)))


if __name__ == "__main__":

    wavelength = torch.tensor(1.55e-6)
    focal_length = torch.tensor(10.e-2)

    input_plane_params = {'name':'plane0', 'center':[0,0,0], 'size':[8.96e-3, 8.96e-3], 'normal_vector': [0,0,1], 'Nx':1080, 'Ny':1080}
    output_plane_params = {'name':'plane1', 'center':[0,0,3.e-2], 'size':[8.96e-3, 8.96e-3], 'normal_vector': [0,0,1], 'Nx':1080, 'Ny':1080}
    
    mod_params = {
            "amplitude_init": 'uniform',
            "phase_init" : 'lens',
            "type" : None,
            "phase_pattern" : lensPhase,
            "amplitude_pattern" : None
        }
 

    params = {
            'input_plane' : input_plane_params,
            'output_plane' : output_plane_params,
            'modulator_params' : mod_params,
            }
    

    db = DiffractiveBlock(params)
