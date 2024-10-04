import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from plane import Plane

from propagators import factory as prop_factory

if __name__ == "__main__":

    params = yaml.load(open("../config.yaml", "r"), Loader=yaml.FullLoader)

    input_plane_params = {
        'name': 'input_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1000,
        'Ny': 1000,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0,0,0])
    }

    output_plane_params0 = {
        'name': 'output_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1000,
        'Ny': 1000,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0,0,11e-2])
    }

    input_plane = Plane(input_plane_params)
    output_plane = Plane(output_plane_params0)

    prop_params = {
            'wavelength':1.55e-6,
            'prop_type' : 'rsc',
            'fft_type' : 'pytorch'}
    factory = prop_factory.PropagatorFactory()

    propagator = factory(input_plane, output_plane, prop_params)

    # Example wavefront to propagate
    # This is a plane wave through a 1mm aperture
    x = torch.linspace(-input_plane.Lx/2, input_plane.Lx/2, input_plane.Nx)
    y = torch.linspace(-input_plane.Ly/2, input_plane.Ly/2, input_plane.Ny)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    wavefront = torch.ones_like(xx)
    wavefront[(xx**2 + yy**2) > (0.6e-3)**2] = 0
    wavefront = wavefront.view(1,input_plane.Nx,input_plane.Ny)
    wavefront = wavefront.type(torch.complex128)

    # Propagate the wavefront
    output_wavefront = propagator(wavefront)

    from IPython import embed; embed()

