# Standard imports
import torch
from loguru import logger
import pytorch_lightning as pl
import sys

# Custom library imports
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))
#from . import plane
#from . import modulator
#from . import propagator
import plane
import modulator
import propagator

class DiffractionBlock(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        input_plane_params = params['input_plane']
        output_plane_params = params['output_plane']
        modulator_params = params['modulator_params']
        propagator_params = params['propagator_params']

        input_plane = plane.Plane(input_plane_params)
        output_plane = plane.Plane(output_plane_params)

        modulator_factory = modulator.ModulatorFactory()
        propagator_factory = propagator.PropagatorFactory()

        self.modulator = modulator_factory(input_plane, modulator_params)
        self.propagator = propagator_factory(input_plane, output_plane, propagator_params)

    def forward(self, input_wavefront=None):
        return(self.propagator(self.modulator(input_wavefront)))


if __name__ == "__main__":

    import yaml 
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

    wavelength = torch.tensor(config['wavelength'])

    focal_length0 = torch.tensor(10.e-2)

    plane0 = plane.Plane(config['planes'][0])
    plane1 = plane.Plane(config['planes'][1])

    lens_phase_pattern0 = modulator.lensPhase(plane1, wavelength, focal_length0)

    propagator_params = config['propagator']

    config['diffraction_blocks'][1]['modulator_params']['phase_pattern'] = lens_phase_pattern0

    db0 = DiffractionBlock(config['diffraction_blocks'][0])
    db1 = DiffractionBlock(config['diffraction_blocks'][1])

    # Example wavefront to propagate
    # This is a plane wave through a 1mm aperture
    xx,yy = plane0.xx, plane0.yy
    wavefront = torch.ones_like(xx)
    wavefront[(xx**2 + yy**2) > (0.2e-3)**2] = 0
    wavefront = wavefront.view(1,1,plane0.Nx,plane0.Ny)

    # Propagate the wavefront
    wavefront0 = db0(wavefront)
    wavefront1 = db1(wavefront0)

    from IPython import embed; embed()

    # Plot the results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,3)
    axs[0].pcolormesh(xx.numpy(),yy.numpy(),wavefront[0,0,:,:].abs().numpy())
    axs[1].pcolormesh(xx.numpy(),yy.numpy(),wavefront0[0,0,:,:].abs().numpy())
    axs[2].pcolormesh(xx.numpy(),yy.numpy(),wavefront1[0,0,:,:].abs().numpy())

    axs[0].set_title('Input')
    axs[1].set_title('Output 0')
    axs[2].set_title('Output 1')

    for ax in axs:
        ax.set_aspect('equal')
    plt.show()
