import torch
from loguru import logger

def initialize_amplitude(params, plane, kwargs:dict={None:None}):
    amplitude_init = params['amplitude_init']
    if amplitude_init == 'uniform':
        logger.info("Uniform amplitude initialization")
        amplitude = uniform_amplitude(plane, kwargs)
    elif amplitude_init == 'random':
        logger.info("Random amplitude initialization")
        amplitude = random_amplitude(plane, kwargs)
    else:
        logger.error("Unsupported amplitude initialization : {}".format(amplitude_init))
        raise Exception('unsupportedInitialization')

    return amplitude

def random_amplitude(plane, kwargs:dict={None:None}) -> torch.Tensor:
    Nx, Ny = plane.Nx, plane.Ny
    amplitude = torch.rand(Nx, Ny)
    if plane.bits == 64:
        amplitude = amplitude.to(torch.float32)
    elif plane.bits == 128:
        amplitude = amplitude.to(torch.float64)
    else:
        logger.error("Invalid number of bits.")
        raise ValueError("Invalid number of bits.")
    return amplitude

def uniform_amplitude(plane, kwargs:dict={None:None}) -> torch.Tensor:
    try:
        value = kwargs['amplitude_value']
    except KeyError:
        logger.warning("Missing value for uniform amplitude initialization. Setting value to 1.0")
        value = 1.0
    Nx, Ny = plane.Nx, plane.Ny
    amplitude = torch.zeros(Nx, Ny) + value
    if plane.bits == 64:
        amplitude = amplitude.to(torch.float32)
    elif plane.bits == 128:
        amplitude = amplitude.to(torch.float64)
    else:
        logger.error("Invalid number of bits.")
        raise ValueError("Invalid number of bits.")
    return amplitude
