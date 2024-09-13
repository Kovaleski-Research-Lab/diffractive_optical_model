import torch
from loguru import logger

def initialize_amplitude(params, plane):
    amplitude_init = params['amplitude_init']
    if amplitude_init == 'uniform':
        logger.info("Uniform amplitude initialization")
        amplitude = uniform_amplitude(plane)
    elif amplitude_init == 'random':
        logger.info("Random amplitude initialization")
        amplitude = random_amplitude(plane)
    else:
        logger.error("Unsupported amplitude initialization : {}".format(amplitude_init))
        raise Exception('unsupportedInitialization')

    return amplitude

def random_amplitude(plane) -> torch.Tensor:
    Nx, Ny = plane.Nx, plane.Ny
    amplitude = torch.rand(Nx, Ny)
    return amplitude.to(torch.float32)

def uniform_amplitude(plane) -> torch.Tensor:
    Nx, Ny = plane.Nx, plane.Ny
    amplitude = torch.ones(Nx, Ny)
    return amplitude.to(torch.float32)
