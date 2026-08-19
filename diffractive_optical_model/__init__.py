from diffractive_optical_model.plane.plane import Plane
from diffractive_optical_model.propagator.factory import PropagatorFactory
from diffractive_optical_model.diffraction_block.diffraction_block import DiffractionBlock

__all__ = ['Plane', 'PropagatorFactory', 'DiffractionBlock', 'DOM']


def __getattr__(name):
    if name == 'DOM':
        try:
            from diffractive_optical_model.diffractive_optical_model import DOM
        except ModuleNotFoundError as exc:
            if exc.name in {'pytorch_lightning', 'torchmetrics'}:
                raise ImportError(
                    "DOM training requires the 'train' extra: "
                    "pip install 'diffractive-optical-model[train]'"
                ) from exc
            raise
        return DOM
    raise AttributeError(name)
