__all__ = ['select_data']


def __getattr__(name):
    if name == 'select_data':
        try:
            from diffractive_optical_model.datamodule.datamodule import select_data
        except ModuleNotFoundError as exc:
            if exc.name == 'torchvision':
                raise ImportError(
                    "MNIST data support requires the 'train' extra: "
                    "pip install 'diffractive-optical-model[train]'"
                ) from exc
            raise
        return select_data
    raise AttributeError(name)
