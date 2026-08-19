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


def initialize_phase(plane, params):
    phase_init = params['phase_init']
    if phase_init == 'uniform':
        logger.info("Uniform phase initialization")
        return uniform_phase(plane, params)
    if phase_init == 'random':
        logger.info("Random phase initialization")
        return random_phase(plane, params)
    if phase_init == 'lens_phase':
        logger.info("Lens phase initialization")
        return lens_phase(plane, params)
    logger.warning("Unsupported phase initialization : {}".format(phase_init))
    raise ValueError("unsupportedInitialization: {}".format(phase_init))


def random_phase(plane, params=None) -> torch.Tensor:
    phase = torch.rand(*_shape(plane)) * (2 * torch.pi)
    return _cast(phase, plane)


def uniform_phase(plane, params=None) -> torch.Tensor:
    if params is None:
        params = {}
    value = params.get('phase_value', 0.0)
    if 'phase_value' not in params:
        logger.warning("Missing value for uniform phase initialization. Setting value to 0.0")
    phase = torch.full(_shape(plane), float(value))
    return _cast(phase, plane)


def lens_phase(plane, params=None) -> torch.Tensor:
    if params is None:
        params = {}
    try:
        focal_length = params['focal_length']
        wavelength = params['wavelength']
    except KeyError as exc:
        logger.error("Missing parameters for lens phase initialization")
        raise ValueError("Missing parameters for lens phase initialization") from exc
    xx, yy = plane.xx, plane.yy
    phase = -(xx ** 2 + yy ** 2) / (2 * focal_length)
    phase = phase * (2 * torch.pi / wavelength)
    f_nyquist = plane.Lx * plane.delta_x / wavelength
    if abs(float(focal_length)) < float(f_nyquist):
        logger.warning(
            "Lens focal length {} may be aliased on this grid "
            "(Nyquist bound |f| >= L Δx / λ = {}).".format(focal_length, float(f_nyquist))
        )
    phase = phase.view(*_shape(plane))
    return _cast(phase, plane)
