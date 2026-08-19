import json
from pathlib import Path

import pytest
import yaml

from diffractive_optical_model.cli import write_run_manifest
from diffractive_optical_model.config import load_config, resolve_config


ROOT = Path(__file__).parents[1]


def _root_config():
    path = ROOT / "config.yaml"
    if path.is_file():
        with path.open() as stream:
            return yaml.safe_load(stream)
    return load_config()[0]


def test_packaged_and_source_default_configs_match():
    source = _root_config()
    loaded, origin = load_config()
    package_path = ROOT / "diffractive_optical_model" / "config.yaml"
    if package_path.is_file():
        with package_path.open() as stream:
            packaged = yaml.safe_load(stream)
        assert packaged == source
    assert loaded["diffraction_blocks"] == source["diffraction_blocks"]
    assert origin.startswith("package:")
    assert Path(loaded["paths"]["path_root"]).is_absolute()


def test_config_schema_rejects_unknown_and_missing_keys():
    config = _root_config()
    config["obsolete_option"] = True
    with pytest.raises(ValueError, match="obsolete_option"):
        resolve_config(config)

    config = _root_config()
    del config["batch_size"]
    with pytest.raises(ValueError, match="batch_size"):
        resolve_config(config)


def test_run_manifest_captures_resolved_research_context(tmp_path):
    config = resolve_config(_root_config(), working_directory=tmp_path)
    destination = write_run_manifest(
        config,
        tmp_path / "run_manifest.json",
        command=["dom-train", "--config", "config.yaml"],
        config_source="config.yaml",
    )
    manifest = json.loads(destination.read_text())
    assert manifest["resolved_config"]["model_id"] == config["model_id"]
    assert manifest["command"] == ["dom-train", "--config", "config.yaml"]
    assert manifest["python"]["version"]
    assert "torch" in manifest["packages"]
    assert "cuda_available" in manifest["hardware"]
    assert set(manifest["git"]) == {"revision", "dirty"}
