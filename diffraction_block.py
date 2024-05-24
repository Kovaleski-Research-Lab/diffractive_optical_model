# Standard imports
import torch
from loguru import logger
import pytorch_lightning as pl
import sys

# Custom library imports
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))
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

    wavelength = torch.tensor(config['propagator']['wavelength'])

    plane0 = plane.Plane(config['planes'][0])
    plane1 = plane.Plane(config['planes'][1])
    plane2 = plane.Plane(config['planes'][2])

    propagator_params = config['propagator']

    db0 = DiffractionBlock(config['diffraction_blocks'][0])
    db1 = DiffractionBlock(config['diffraction_blocks'][1])

    # Example wavefront to propagate
    # This is a plane wave through a 1mm aperture
    x,y = plane0.x, plane0.y
    xx,yy = plane0.xx, plane0.yy
    wavefront = torch.ones_like(xx)
    wavefront[(xx.real**2 + yy.real**2) > (1.e-3)**2] = 0
    wavefront = wavefront.view(1,1,plane0.Nx,plane0.Ny)

    # Propagate the wavefront
    wavefront0 = db0(wavefront)
    wavefront1 = db1(wavefront0).detach()

    import matplotlib.pyplot as plt
    from IPython import embed; embed()

    x_in,y_in = plane0.x, plane0.y
    xx_in,yy_in = plane0.xx, plane0.yy
    x_lens,y_lens = plane1.x, plane1.y
    xx_lens,yy_lens = plane1.xx, plane1.yy
    x_out,y_out = plane2.x, plane2.y
    xx_out,yy_out = plane2.xx, plane2.yy
    # Plot the results
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(wavefront.T.squeeze().abs().numpy())
    axs[1].imshow(wavefront0.T.squeeze().abs().numpy())
    axs[2].imshow(wavefront1.T.squeeze().abs().numpy())

    axs[0].set_title('Input')
    axs[1].set_title('Output 0')
    axs[2].set_title('Output 1')

    for ax in axs:
        ax.set_aspect('equal')
    plt.show()
