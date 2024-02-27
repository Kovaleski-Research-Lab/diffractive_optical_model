
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from propagator import *
from plane import *


def test_zoom_in_asm():

    input_plane_params = {
        'name': 'input_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0,0,0])
    }

    output_plane_params0 = {
        'name': 'output_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0.,0., 9e-2])
    }

    output_plane_params1 = {
        'name': 'output_plane',
        'size': torch.tensor([2e-3, 2e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0.,0.,9e-2])
    }

    input_plane = plane.Plane(input_plane_params)
    output_plane0 = plane.Plane(output_plane_params0)
    output_plane1 = plane.Plane(output_plane_params1)

    propagator_params = {
        'prop_type': None,
        'wavelength': torch.tensor(1.55e-6),
        'czt': False
    }

    propagator0 = PropagatorFactory()(input_plane, output_plane0, propagator_params)
    propagator_params['czt'] = True
    propagator1 = PropagatorFactory()(input_plane, output_plane1, propagator_params)


    input_wavefront = torch.ones_like(input_plane.xx)
    input_wavefront[(input_plane.xx**2 + input_plane.yy**2) > (0.2e-3)**2] = 0
    input_wavefront = input_wavefront.view(1,1,input_plane.Nx,input_plane.Ny)

    output_wavefront0 = propagator0(input_wavefront)
    output_wavefront1 = propagator1(input_wavefront)

    fig,ax = plt.subplots(1,3)
    ax[0].imshow(input_wavefront[0,0].abs().numpy())
    ax[1].imshow(output_wavefront0[0,0].abs().numpy())
    ax[2].imshow(output_wavefront1[0,0].abs().numpy())
    plt.show()

def test_zoom_out_asm():

    input_plane_params = {
        'name': 'input_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0,0,0])
    }

    output_plane_params0 = {
        'name': 'output_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0.,0., 9e-2])
    }

    output_plane_params1 = {
        'name': 'output_plane',
        'size': torch.tensor([50e-3, 50e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0.,0.,9e-2])
    }

    input_plane = plane.Plane(input_plane_params)
    output_plane0 = plane.Plane(output_plane_params0)
    output_plane1 = plane.Plane(output_plane_params1)

    propagator_params = {
        'prop_type': None,
        'wavelength': torch.tensor(1.55e-6),
        'czt': False
    }

    propagator0 = PropagatorFactory()(input_plane, output_plane0, propagator_params)
    propagator_params['czt'] = True
    propagator1 = PropagatorFactory()(input_plane, output_plane1, propagator_params)

    input_wavefront = torch.ones_like(input_plane.xx)
    input_wavefront[(input_plane.xx**2 + input_plane.yy**2) > (0.2e-3)**2] = 0
    input_wavefront = input_wavefront.view(1,1,input_plane.Nx,input_plane.Ny)

    output_wavefront0 = propagator0(input_wavefront)
    output_wavefront1 = propagator1(input_wavefront)

    fig,ax = plt.subplots(1,3)
    ax[0].imshow(input_wavefront[0,0].abs().numpy())
    ax[1].imshow(output_wavefront0[0,0].abs().numpy())
    ax[2].imshow(output_wavefront1[0,0].abs().numpy())
    plt.show()


def test_zoom_in_rsc():
    input_plane_params = {
        'name': 'input_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0,0,0])
    }

    output_plane_params0 = {
        'name': 'output_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0.,0.,10e-2])
    }

    output_plane_params1 = {
        'name': 'output_plane',
        'size': torch.tensor([2e-3, 2e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0.,0.,10e-2])
    }

    input_plane = plane.Plane(input_plane_params)
    output_plane0 = plane.Plane(output_plane_params0)
    output_plane1 = plane.Plane(output_plane_params1)

    propagator_params = {
        'prop_type': None,
        'wavelength': torch.tensor(1.55e-6),
        'czt': False
    }

    propagator0 = PropagatorFactory()(input_plane, output_plane0, propagator_params)
    propagator_params['czt'] = True
    propagator1 = PropagatorFactory()(input_plane, output_plane1, propagator_params)



    input_wavefront = torch.ones_like(input_plane.xx)
    input_wavefront[(input_plane.xx**2 + input_plane.yy**2) > (0.2e-3)**2] = 0
    input_wavefront = input_wavefront.view(1,1,input_plane.Nx,input_plane.Ny)

    output_wavefront0 = propagator0(input_wavefront)
    output_wavefront1 = propagator1(input_wavefront)

    fig,ax = plt.subplots(1,3)
    ax[0].imshow(input_wavefront[0,0].abs().numpy())
    ax[1].imshow(output_wavefront0[0,0].abs().numpy())
    ax[2].imshow(output_wavefront1[0,0].abs().numpy())
    plt.show()



def test_zoom_out_rsc():
    input_plane_params = {
        'name': 'input_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0,0,0])
    }

    output_plane_params0 = {
        'name': 'output_plane',
        'size': torch.tensor([8.96e-3, 8.96e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0.,0.,20e-2])
    }

    output_plane_params1 = {
        'name': 'output_plane',
        'size': torch.tensor([50e-3, 50e-3]),
        'Nx': 1080,
        'Ny': 1080,
        'normal': torch.tensor([0,0,1]),
        'center': torch.tensor([0.,0.,20e-2])
    }

    input_plane = plane.Plane(input_plane_params)
    output_plane0 = plane.Plane(output_plane_params0)
    output_plane1 = plane.Plane(output_plane_params1)

    propagator_params = {
        'prop_type': None,
        'wavelength': torch.tensor(1.55e-6),
        'czt': False
    }

    propagator0 = PropagatorFactory()(input_plane, output_plane0, propagator_params)
    propagator_params['czt'] = True
    propagator1 = PropagatorFactory()(input_plane, output_plane1, propagator_params)



    input_wavefront = torch.ones_like(input_plane.xx)
    input_wavefront[(input_plane.xx**2 + input_plane.yy**2) > (0.2e-3)**2] = 0
    input_wavefront = input_wavefront.view(1,1,input_plane.Nx,input_plane.Ny)

    output_wavefront0 = propagator0(input_wavefront)
    output_wavefront1 = propagator1(input_wavefront)

    fig,ax = plt.subplots(1,3)
    ax[0].imshow(input_wavefront[0,0].abs().numpy())
    ax[1].imshow(output_wavefront0[0,0].abs().numpy())
    ax[2].imshow(output_wavefront1[0,0].abs().numpy())
    plt.show()



if __name__ == "__main__":

    test_zoom_in_rsc()
    test_zoom_out_rsc()
    test_zoom_in_asm()
    test_zoom_out_asm()

