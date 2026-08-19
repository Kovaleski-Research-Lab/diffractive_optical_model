import torch
from loguru import logger


def _shape(plane):
    return 1, 1, int(plane.Nx), int(plane.Ny)


def _cast(tensor, plane):
    if plane.bits == 64:
        return tensor.to(torch.float32)
    if plane.bits == 128:
        return tensor.to(torch.float64)
    raise ValueError("Invalid number of bits.")


def initialize_amplitude(plane, params):
    amplitude_init = params['amplitude_init']
    if amplitude_init == 'uniform':
        logger.info("Uniform amplitude initialization")
        return uniform_amplitude(plane, params)
    if amplitude_init == 'random':
        logger.info("Random amplitude initialization")
        return random_amplitude(plane, params)
    if amplitude_init == 'pinhole':
        logger.info("Pinhole amplitude initialization")
        return pinhole(plane, params)
    logger.error("Unsupported amplitude initialization : {}".format(amplitude_init))
    raise ValueError("unsupportedInitialization: {}".format(amplitude_init))


def random_amplitude(plane, params=None) -> torch.Tensor:
    amplitude = torch.rand(*_shape(plane))
    return _cast(amplitude, plane)


def uniform_amplitude(plane, params=None) -> torch.Tensor:
    if params is None:
        params = {}
    value = params.get('amplitude_value', 1.0)
    if 'amplitude_value' not in params:
        logger.warning("Missing value for uniform amplitude initialization. Setting value to 1.0")
    amplitude = torch.full(_shape(plane), float(value))
    return _cast(amplitude, plane)


def pinhole(plane, params=None) -> torch.Tensor:
    if params is None:
        params = {}
    pinhole_size = float(params['pinhole_size'])
    amplitude = torch.zeros(*_shape(plane))
    xx, yy = plane.xx, plane.yy
    mask = (xx ** 2 + yy ** 2) < pinhole_size ** 2
    amplitude[0, 0][mask] = 1.0
    return _cast(amplitude, plane)
