"""Installed training CLI and run-provenance capture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch

from diffractive_optical_model.config import load_config


_DISTRIBUTIONS = (
    "diffractive-optical-model",
    "loguru",
    "numpy",
    "pytorch-lightning",
    "PyYAML",
    "torch",
    "torchmetrics",
    "torchvision",
)


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _git_state(directory: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(directory), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    revision = git("rev-parse", "HEAD")
    if revision is None:
        return {"revision": None, "dirty": None}
    status = git("status", "--porcelain")
    return {"revision": revision, "dirty": bool(status) if status is not None else None}


def _hardware_state() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": [properties.major, properties.minor],
                }
            )

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": devices,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def build_run_manifest(
    config: Mapping[str, Any],
    *,
    command: Sequence[str] | None = None,
    config_source: str | None = None,
) -> dict[str, Any]:
    """Collect the environment needed to interpret a training run."""

    command_parts = list(command if command is not None else sys.argv)
    seed_setting = config.get("seed")
    paths = config.get("paths")
    git_directory = Path.cwd()
    if isinstance(paths, Mapping) and paths.get("path_root"):
        git_directory = Path(str(paths["path_root"])).expanduser()
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command_parts,
        "command_display": shlex.join(command_parts),
        "config_source": config_source,
        "resolved_config": dict(config),
        "seed": seed_setting,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            distribution: _version(distribution) for distribution in _DISTRIBUTIONS
        },
        "git": _git_state(git_directory),
        "hardware": _hardware_state(),
    }


def write_run_manifest(
    config: Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    command: Sequence[str] | None = None,
    config_source: str | None = None,
) -> Path:
    """Atomically write a JSON run manifest and return its final path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config, command=command, config_source=config_source
    )

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(manifest, temporary, indent=2, sort_keys=True, default=str)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dom-train",
        description="Train a diffractive optical model from a YAML configuration.",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="YAML configuration path (defaults to the packaged config resource).",
    )
    parser.add_argument(
        "--manifest",
        help="Override the run-manifest path derived from paths.path_results.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_version('diffractive-optical-model') or '0.0.1'}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        config, source = load_config(arguments.config)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    command = [sys.argv[0], *(list(argv) if argv is not None else sys.argv[1:])]
    try:
        from train import run
    except ModuleNotFoundError as exc:
        parser.error(
            f"Training dependency {exc.name!r} is unavailable; install "
            "'diffractive-optical-model[train]'."
        )

    run(
        config,
        command=command,
        config_source=source,
        manifest_path=arguments.manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
