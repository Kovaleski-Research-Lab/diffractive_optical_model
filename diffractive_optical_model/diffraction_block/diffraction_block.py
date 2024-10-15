# Standard imports
import torch
from loguru import logger
import pytorch_lightning as pl
import sys

# Custom library imports
import os

from diffractive_optical_model.plane.plane import Plane
from diffractive_optical_model.modulator.factory import ModulatorFactory
from diffractive_optical_model.propagator.factory import PropagatorFactory
from diffractive_optical_model.utils.spatial_resample import spatial_resample

class DiffractionBlock(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        self.input_plane_params = params['input_plane']
        self.output_plane_params = params['output_plane']
        self.modulator_params = params['modulator_params']
        self.propagator_params = params['propagator_params']

        self.input_plane = Plane(self.input_plane_params, bits=128)
        self.output_plane = Plane(self.output_plane_params, bits=128)

        self.modulator = ModulatorFactory()(self.input_plane, self.modulator_params)
        self.propagator = PropagatorFactory()(self.input_plane, self.output_plane, self.propagator_params)

    def forward(self, input_wavefront=None):
        modulated_wavefront = self.modulator(input_wavefront)
        propagated_wavefront = self.propagator(modulated_wavefront)
        return propagated_wavefront


if __name__ == "__main__":

    import yaml 
    config = yaml.load(open('../../config.yaml'), Loader=yaml.FullLoader)

    wavelength = torch.tensor(config['propagator']['wavelength'])

    plane0 = Plane(config['planes'][0])
    plane1 = Plane(config['planes'][1])
    plane2 = Plane(config['planes'][2])

    propagator_params = config['propagator']

    db0 = DiffractionBlock(config['diffraction_blocks'][0])
    db1 = DiffractionBlock(config['diffraction_blocks'][1])

    # Example wavefront to propagate
    # This is a plane wave through a 1mm aperture
    x,y = plane0.x, plane0.y
    xx,yy = plane0.xx, plane0.yy
    wavefront = torch.ones_like(xx)
    wavefront[(xx.real**2 + yy.real**2) > (1)**2] = 0
    wavefront = wavefront.view(1,1,plane0.Nx,plane0.Ny)
    wavefront = wavefront.type(torch.complex64)
    
    # Propagate the wavefront
    wavefront0 = db0.forward(wavefront)
    wavefront1 = db1.forward(wavefront0)

    from IPython import embed; embed()

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Plot the results
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(wavefront.T.squeeze().abs().numpy()**2)
    axs[1].imshow(wavefront0.T.squeeze().abs().numpy()**2)
    axs[2].imshow(wavefront1.T.squeeze().abs().numpy()**2)

    axs[0].set_title('Input')
    axs[1].set_title('Output 0')
    axs[2].set_title('Output 1')

    for ax in axs:
        ax.set_aspect('equal')
    plt.show()


    scale = 0.6
    scaled_plane = plane0.scale(scale, inplace=False)

    resampled_wavefront = spatial_resample(scaled_plane, wavefront.abs(), plane2)
    image = wavefront1.abs().squeeze().numpy()**2
    image = image/image.max()

    # Plot the resampled wavefront, wavefront1, and their difference using pcolormesh
    fig, axs = plt.subplots(1,3, figsize=(15,5))

    im0 = axs[0].pcolormesh(plane2.xx, plane2.yy, resampled_wavefront.squeeze().abs().numpy()**2)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax=cax)
    axs[0].set_title("Resampled wavefront")
    axs[0].set_aspect('equal')

    im1 = axs[1].pcolormesh(plane2.xx, plane2.yy, image)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)
    axs[1].set_title("Wavefront 1")
    axs[1].set_aspect('equal')

    im2 = axs[2].pcolormesh(plane2.xx, plane2.yy, (resampled_wavefront.squeeze().abs()**2 - image))
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax)
    axs[2].set_title("Difference")
    axs[2].set_aspect('equal')

    for ax in axs:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

    plt.tight_layout()
    plt.show()
