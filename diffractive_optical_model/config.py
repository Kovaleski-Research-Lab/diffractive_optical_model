"""Configuration loading for source checkouts and installed distributions."""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_NAME = "config.yaml"

_TOP_LEVEL_KEYS = {
    "seed", "model_id", "bits", "batch_size", "num_epochs", "gpu_config",
    "valid_rate", "paths", "diffraction_blocks", "dom_training", "which",
    "n_cpus", "resize_row", "resize_col", "Nxp", "Nyp", "wavefront_transform",
}
_REQUIRED_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS
_PATH_KEYS = {"path_root", "path_data", "path_results", "path_checkpoint"}
_TRAINING_KEYS = {"optimizer", "learning_rate", "objective_function", "data_range"}
_BLOCK_KEYS = {
    "input_plane", "output_plane", "modulator_params", "propagator_params", "bits",
}
_PLANE_KEYS = {"name", "center", "size", "normal", "Nx", "Ny"}
_MODULATOR_KEYS = {
    "gradients", "amplitude_init", "amplitude_value", "pinhole_size",
    "phase_init", "phase_value", "focal_length", "wavelength",
}
_PROPAGATOR_KEYS = {
    "wavelength", "fft_type", "prop_type", "padded", "bits",
    "validate_sampling", "allow_aliasing",
}
_WAVEFRONT_KEYS = {"phase_initialization_strategy", "bits"}


def _packaged_config_text(name: str = DEFAULT_CONFIG_NAME) -> tuple[str, str]:
    resource = resources.files("diffractive_optical_model").joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(
            f"Packaged default configuration {name!r} is unavailable. "
            "Pass --config PATH. Distributors may provide the default as "
            f"diffractive_optical_model/{name}."
        )
    return resource.read_text(encoding="utf-8"), f"package:{resource}"


def _check_keys(mapping, allowed, required, label):
    if not isinstance(mapping, Mapping):
        raise TypeError(f"Configuration section {label!r} must be a mapping.")
    unknown = sorted(str(key) for key in set(mapping).difference(allowed))
    if unknown:
        raise ValueError(
            f"Unknown configuration keys in {label}: {', '.join(unknown)}"
        )
    missing = sorted(str(key) for key in set(required).difference(mapping))
    if missing:
        raise ValueError(
            f"Missing required configuration keys in {label}: {', '.join(missing)}"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the supported configuration surface and reject stale keys."""
    _check_keys(
        config,
        _TOP_LEVEL_KEYS,
        _REQUIRED_TOP_LEVEL_KEYS,
        "root",
    )
    _check_keys(config["paths"], _PATH_KEYS, _PATH_KEYS, "paths")
    _check_keys(
        config["dom_training"],
        _TRAINING_KEYS,
        _TRAINING_KEYS,
        "dom_training",
    )
    _check_keys(
        config["wavefront_transform"],
        _WAVEFRONT_KEYS,
        {"phase_initialization_strategy"},
        "wavefront_transform",
    )
    blocks = config["diffraction_blocks"]
    if not isinstance(blocks, Mapping) or not blocks:
        raise ValueError("diffraction_blocks must be a non-empty mapping.")
    for block_id, block in blocks.items():
        label = f"diffraction_blocks.{block_id}"
        _check_keys(
            block,
            _BLOCK_KEYS,
            {"input_plane", "output_plane", "modulator_params", "propagator_params"},
            label,
        )
        _check_keys(
            block["input_plane"], _PLANE_KEYS, _PLANE_KEYS, f"{label}.input_plane"
        )
        _check_keys(
            block["output_plane"], _PLANE_KEYS, _PLANE_KEYS, f"{label}.output_plane"
        )
        _check_keys(
            block["modulator_params"],
            _MODULATOR_KEYS,
            {"gradients", "amplitude_init", "phase_init"},
            f"{label}.modulator_params",
        )
        _check_keys(
            block["propagator_params"],
            _PROPAGATOR_KEYS,
            {"wavelength", "fft_type", "prop_type", "padded"},
            f"{label}.propagator_params",
        )

    if config["bits"] not in (64, 128):
        raise ValueError("bits must be 64 or 128.")
    for key in ("batch_size", "num_epochs", "Nxp", "Nyp", "resize_row", "resize_col"):
        value = config[key]
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"{key} must be a strictly positive integer.")
    if str(config["which"]).upper() != "MNIST":
        raise ValueError("The only supported dataset is MNIST.")


def resolve_config(
    config: Mapping[str, Any],
    *,
    working_directory: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a detached config with an absolute, expanded ``path_root``."""

    resolved = deepcopy(dict(config))
    validate_config(resolved)
    paths = resolved.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("Configuration key 'paths' must be a mapping.")

    base = Path(working_directory or Path.cwd()).expanduser().resolve()
    configured_root = os.path.expandvars(str(paths.get("path_root") or "."))
    root = Path(configured_root).expanduser()
    if not root.is_absolute():
        root = base / root
    paths["path_root"] = str(root.resolve())
    return resolved


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    working_directory: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Load an explicit YAML file or the package's future default resource."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "YAML configuration support requires the training extra: "
            "pip install 'diffractive-optical-model[train]'"
        ) from exc

    if path is None:
        text, source = _packaged_config_text()
    else:
        config_path = Path(path).expanduser().resolve(strict=True)
        text = config_path.read_text(encoding="utf-8")
        source = str(config_path)

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise TypeError("The training configuration must contain a YAML mapping.")
    return resolve_config(loaded, working_directory=working_directory), source
