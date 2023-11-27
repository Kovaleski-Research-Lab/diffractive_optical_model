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
    focal_length0 = torch.tensor(10.e-2)
    focal_length1 = torch.tensor(20.e-2)

    Nx = 1920
    Ny = 1920

    # Input plane
    plane0_params = {'name':'plane0', 'center':[0,0,0], 'size':[8.96e-3, 8.96e-3], 'normal': [0,0,1], 'Nx':Nx, 'Ny':Ny}

    # Lens 0 plane
    plane1_params = {'name':'plane1', 'center':[0,0,20e-2], 'size':[8.96e-3, 8.96e-3], 'normal': [0,0,1], 'Nx':Nx, 'Ny':Ny}

    # Image 0 plane
    plane2_params = {'name':'plane2', 'center':[0,0,30e-2], 'size':[8.96e-3, 8.96e-3], 'normal': [0,0,1], 'Nx':Nx, 'Ny':Ny}

    # Lens 1 plane
    plane3_params = {'name':'plane3', 'center':[0,0,50e-2], 'size':[8.96e-3, 8.96e-3], 'normal': [0,0,1], 'Nx':Nx, 'Ny':Ny}

    # Image 1 plane
    plane4_params = {'name':'plane4', 'center':[0,0,55e-2], 'size':[8.96e-3, 8.96e-3], 'normal': [0,0,1], 'Nx':Nx, 'Ny':Ny}


    plane0 = Plane(plane0_params)
    plane1 = Plane(plane1_params)
    plane2 = Plane(plane2_params)
    plane3 = Plane(plane3_params)
    plane4 = Plane(plane4_params)

    lens_phase_pattern0 = lensPhase(plane1, wavelength, focal_length0)
    lens_phase_pattern1 = lensPhase(plane3, wavelength, focal_length1)
    
    source_mod_params = {
            "amplitude_init": 'uniform',
            "phase_init" : 'uniform',
            "type" : None,
            "phase_pattern" : None,
            "amplitude_pattern" : None
        }
 
    lens0_mod_params = {
            "amplitude_init": 'uniform',
            "phase_init" : 'custom',
            "type" : None,
            "phase_pattern" : lens_phase_pattern0,
            "amplitude_pattern" : None
        }

    lens1_mod_params = {
            "amplitude_init": 'uniform',
            "phase_init" : 'custom',
            "type" : None,
            "phase_pattern" : lens_phase_pattern1,
            "amplitude_pattern" : None
        }

    propagator_params = {
            "wavelength" : wavelength,
        }



    #Source
    params_block0 = {
            'input_plane' : plane0_params,
            'output_plane' : plane1_params,
            'modulator_params' : source_mod_params,
            'propagator_params' : propagator_params
            }
    
    #Lens 0
    params_block1 = {
            'input_plane' : plane1_params,
            'output_plane' : plane2_params,
            'modulator_params' : lens0_mod_params,
            'propagator_params' : propagator_params
            }
    
    #Image 0
    params_block2 = {
            'input_plane' : plane2_params,
            'output_plane' : plane3_params,
            'modulator_params' : source_mod_params,
            'propagator_params' : propagator_params
            }

    #Lens 1
    params_block3 = {
            'input_plane' : plane3_params,
            'output_plane' : plane4_params,
            'modulator_params' : lens1_mod_params,
            'propagator_params' : propagator_params
            }

    db0 = DiffractiveBlock(params_block0)
    db1 = DiffractiveBlock(params_block1)
    db2 = DiffractiveBlock(params_block2)
    db3 = DiffractiveBlock(params_block3)


    # Example wavefront to propagate
    # This is a plane wave through a 1mm aperture
    xx,yy = plane0.xx, plane0.yy
    wavefront = torch.ones_like(xx)
    wavefront[(xx**2 + yy**2) > (0.2e-3)**2] = 0
    wavefront = wavefront.view(1,1,plane0.Nx,plane0.Ny)

    # Propagate the wavefront
    wavefront0 = db0(wavefront)
    wavefront1 = db1(wavefront0)
    wavefront2 = db2(wavefront1)
    wavefront3 = db3(wavefront2)

    # Plot the results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,5)
    axs[0].pcolormesh(xx,yy,wavefront[0,0,:,:].abs().numpy())
    axs[1].pcolormesh(xx,yy,wavefront0[0,0,:,:].abs().numpy())
    axs[2].pcolormesh(xx,yy,wavefront1[0,0,:,:].abs().numpy())
    axs[3].pcolormesh(xx,yy,wavefront2[0,0,:,:].abs().numpy())
    axs[4].pcolormesh(xx,yy,wavefront3[0,0,:,:].abs().numpy())

    axs[0].set_title('Input')
    axs[1].set_title('Output 0')
    axs[2].set_title('Output 1')
    axs[3].set_title('Output 2')
    axs[4].set_title('Output 3')

    for ax in axs:
        ax.set_aspect('equal')

    plt.show()
