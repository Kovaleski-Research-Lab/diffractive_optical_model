import torch
from loguru import logger

def initialize_phase(params, plane):
    phase_init = params['phase_init']

    if phase_init == 'uniform':
        logger.info("Uniform phase initialization")
        phase = uniform_phase(plane)
    elif phase_init == 'random':
        logger.info("Random phase initialization")
        phase = random_phase(plane)
    else:
        logger.warning("Unsupported phase initialization : {}".format(phase_init))
        logger.warning("Setting uniform phase initialization")
        phase = uniform_phase(plane)
        raise Exception('unsupportedInitialization')

    return phase

def random_phase(plane) -> torch.Tensor:
    Nx, Ny = plane.Nx, plane.Ny
    phase = torch.rand(1,1,Nx, Ny)
    return phase.to(torch.float32)

def uniform_phase(plane) -> torch.Tensor:
    Nx, Ny = plane.Nx, plane.Ny
    phase = torch.zeros(Nx, Ny)
    return phase.to(torch.float32)
