import torch
from loguru import logger

def initialize_phase(params, plane, kwargs:dict={None:None}):
    phase_init = params['phase_init']

    if phase_init == 'uniform':
        logger.info("Uniform phase initialization")
        phase = uniform_phase(plane, kwargs)
    elif phase_init == 'random':
        logger.info("Random phase initialization")
        phase = random_phase(plane, kwargs)
    elif phase_init == 'lens_phase':
        logger.info("Lens phase initialization")
        phase = lens_phase(plane, kwargs)
    else:
        logger.warning("Unsupported phase initialization : {}".format(phase_init))
        logger.warning("Setting uniform phase initialization")
        phase = uniform_phase(plane, kwargs)
        raise Exception('unsupportedInitialization')

    return phase

def random_phase(plane, kwargs:dict={None:None}) -> torch.Tensor:
    Nx, Ny = plane.Nx, plane.Ny
    phase = torch.rand(1,1,Nx, Ny)
    if plane.bits == 64:
        phase = phase.to(torch.float32)
    elif plane.bits == 128:
        phase = phase.to(torch.float64)
    else:
        logger.error("Invalid number of bits.")
        raise ValueError("Invalid number of bits.")
    return phase

def uniform_phase(plane, kwargs:dict={None:None}) -> torch.Tensor:
    try:
        value = kwargs['phase_value']
    except KeyError:
        logger.warning("Missing value for uniform phase initialization. Setting value to 0.0")
        value = 0.0
    Nx, Ny = plane.Nx, plane.Ny
    phase = torch.zeros(Nx, Ny) + value
    if plane.bits == 64:
        phase = phase.to(torch.float32)
    elif plane.bits == 128:
        phase = phase.to(torch.float64)
    else:
        logger.error("Invalid number of bits.")
        raise ValueError("Invalid number of bits.")
    return phase

def lens_phase(plane, kwargs:dict={None:None}) -> torch.Tensor:
    try:
        focal_length = kwargs['focal_length']
        wavelength = kwargs['wavelength']
    except KeyError:
        logger.error("Missing parameters for lens phase initialization")
        raise ValueError("Missing parameters for lens phase initialization")
    xx,yy = plane.xx, plane.yy
    Nx, Ny = plane.Nx, plane.Ny
    phase = -(xx**2 + yy**2) / ( 2 * focal_length )
    phase *= (2 * torch.pi / wavelength)
    phase = phase.view(1,1,Nx,Ny)
    if plane.bits == 64:
        phase = phase.to(torch.float32)
    elif plane.bits == 128:
        phase = phase.to(torch.float64)
    else:
        logger.error("Invalid number of bits.")
        raise ValueError("Invalid number of bits.")
    return phase

