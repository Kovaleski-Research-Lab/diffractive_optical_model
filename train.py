import os
from pathlib import Path
import sys
from typing import Sequence

import yaml
import torch
import logging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from diffractive_optical_model.cli import write_run_manifest
from diffractive_optical_model.config import resolve_config
from diffractive_optical_model.diffractive_optical_model import DOM
from diffractive_optical_model.datamodule.datamodule import select_data


def run(
    params,
    *,
    command: Sequence[str] | None = None,
    config_source: str | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
):
    logging.debug("train.py() | running training")

    params = resolve_config(params)

    paths = params['paths']
    path_root = paths['path_root']
    path_checkpoint = paths['path_checkpoint']
    path_results = paths['path_results']
    model_id = params['model_id']

    dump_dir = os.path.join(path_root, path_results, model_id)
    if manifest_path is None:
        manifest_path = os.path.join(dump_dir, 'run_manifest.json')
    written_manifest = write_run_manifest(
        params,
        manifest_path,
        command=command,
        config_source=config_source,
    )
    logging.info("Run manifest written to %s", written_manifest)

    if params['seed'][0]:
        seed_everything(params['seed'][1], workers=True)

    model = DOM(params)
    data = select_data(params)

    checkpoint_path = os.path.join(path_root, path_results, path_checkpoint, model_id)
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path)
    logging.debug(f'Checkpoint path: {checkpoint_path}')

    logging.debug('Setting matmul precision to HIGH')
    torch.set_float32_matmul_precision('high')

    gpu_list = params['gpu_config'][1]
    num_epochs = params['num_epochs']

    if params['gpu_config'][0] and torch.cuda.is_available():
        logging.debug("Training with GPUs")
        trainer = Trainer(
            accelerator="cuda",
            num_nodes=1,
            num_sanity_val_steps=0,
            devices=gpu_list,
            max_epochs=num_epochs,
            deterministic=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            default_root_dir=path_root,
            callbacks=[checkpoint_callback],
        )
    else:
        logging.debug("Training with CPUs")
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=num_epochs,
            num_sanity_val_steps=0,
            default_root_dir=os.path.join(path_root, path_results),
            check_val_every_n_epoch=1,
            callbacks=[checkpoint_callback],
        )

    trainer.fit(model, data)

    os.makedirs(dump_dir, exist_ok=True)
    with open(os.path.join(dump_dir, 'params.yaml'), 'w', encoding='utf-8') as stream:
        yaml.safe_dump(params, stream, sort_keys=True)


if __name__ == "__main__":
    from diffractive_optical_model.cli import main

    cli_arguments = sys.argv[1:]
    source_default = Path(__file__).with_name("config.yaml")
    has_explicit_config = any(
        argument == "--config"
        or argument == "-c"
        or argument.startswith("--config=")
        for argument in cli_arguments
    )
    if source_default.is_file() and not has_explicit_config:
        cli_arguments = ["--config", str(source_default), *cli_arguments]
    raise SystemExit(main(cli_arguments))
