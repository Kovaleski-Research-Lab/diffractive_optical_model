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
        propagator_params = params['propagator_params']

        input_plane = Plane(input_plane_params)
        output_plane = Plane(output_plane_params)

        modulator_factory = ModulatorFactory()
        propagator_factory = PropagatorFactory()

        self.modulator = modulator_factory(input_plane, modulator_params)
        self.propagator = propagator_factory(input_plane, output_plane, propagator_params)

    def forward(self, input_wavefront):
        return(self.propagator(self.modulator(input_wavefront)))


if __name__ == "__main__":

    wavelength = torch.tensor(1.55e-6)
    focal_length = torch.tensor(10.e-2)

    input_plane0_params = {'name':'plane0', 'center':[0,0,0], 'size':[8.96e-3, 8.96e-3], 'normal': [0,0,1], 'Nx':1080, 'Ny':1080}
    output_plane0_params = {'name':'plane1', 'center':[0,0,20.e-2], 'size':[8.96e-3, 8.96e-3], 'normal': [0,0,1], 'Nx':1080, 'Ny':1080}
    output_plane1_params = {'name':'plane3', 'center':[0,0,40.e-2], 'size':[8.96e-3, 8.96e-3], 'normal': [0,0,1], 'Nx':1080, 'Ny':1080}

    input_plane0 = Plane(input_plane0_params)
    output_plane0 = Plane(output_plane0_params)
    output_plane1 = Plane(output_plane1_params)

    lens_phase_pattern = lensPhase(output_plane0, wavelength, focal_length)
    
    source_mod_params = {
            "amplitude_init": 'uniform',
            "phase_init" : 'uniform',
            "type" : None,
            "phase_pattern" : lens_phase_pattern,
            "amplitude_pattern" : None
        }
 
    lens_mod_params = {
            "amplitude_init": 'uniform',
            "phase_init" : 'custom',
            "type" : None,
            "phase_pattern" : lens_phase_pattern,
            "amplitude_pattern" : None
        }

    propagator_params = {
            "wavelength" : wavelength,
        }

    params_block0 = {
            'input_plane' : input_plane0_params,
            'output_plane' : output_plane0_params,
            'modulator_params' : source_mod_params,
            'propagator_params' : propagator_params
            }
    
    params_block1 = {
            'input_plane' : output_plane0_params,
            'output_plane' : output_plane1_params,
            'modulator_params' : lens_mod_params,
            'propagator_params' : propagator_params
            }

    db0 = DiffractiveBlock(params_block0)
    db1 = DiffractiveBlock(params_block1)

    # Example wavefront to propagate
    # This is a plane wave through a 1mm aperture
    xx,yy = input_plane0.xx, input_plane0.yy
    wavefront = torch.ones_like(xx)
    wavefront[(xx**2 + yy**2) > (1e-3)**2] = 0
    wavefront = wavefront.view(1,1,input_plane0.Nx,input_plane0.Ny)

    # Propagate the wavefront
    wavefront0 = db0(wavefront)
    wavefront1 = db1(wavefront0)

    # Plot the results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,3)
    axs[0].pcolormesh(xx,yy,wavefront[0,0,:,:].abs().numpy())
    axs[1].pcolormesh(xx,yy,wavefront0[0,0,:,:].abs().numpy())
    axs[2].pcolormesh(xx,yy,wavefront1[0,0,:,:].abs().numpy())

    axs[0].set_title('Input')
    axs[1].set_title('Output 0')
    axs[2].set_title('Output 1')

    for ax in axs:
        ax.set_aspect('equal')

    plt.show()
