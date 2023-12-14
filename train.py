#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import logging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

import datamodule
import don

#--------------------------------
# Initialize: Training
#--------------------------------

def run(params):
    logging.debug("train.py() | running training")
    #logging.basicConfig(level=logging.DEBUG)
          
    # Initialize: Seeding
    if params['seed'][0]:
        seed_everything(params['seed'][1], workers = True)

    # Initialize: The model
    model = don.DON(params)

    # Initialize: The datamodule
    data = datamodule.select_data(params)

    # Initialize:  PytorchLighting model checkpoint
    paths = params['paths']
    path_root = paths['path_root']
    path_checkpoint = paths['path_checkpoint']
    path_results = paths['path_results']

    model_id = params['model_id']

    checkpoint_path = os.path.join(path_root, path_results, path_checkpoint, model_id)
    checkpoint_callback = ModelCheckpoint(dirpath = checkpoint_path)
    logging.debug(f'Checkpoint path: {checkpoint_path}')

    logging.debug('Setting matmul precision to HIGH')
    torch.set_float32_matmul_precision('high')

    gpu_list = params['gpu_config'][1]
    num_epochs = params['num_epochs']
    path_results = paths['path_results']

    # Initialize: PytorchLightning Trainer
    if(params['gpu_config'][0] and torch.cuda.is_available()):
        logging.debug("Training with GPUs")
        trainer = Trainer(accelerator = "cuda", num_nodes = 1, 
                          check_val_every_n_epoch = 1, num_sanity_val_steps = 1,
                          devices = gpu_list, max_epochs = num_epochs, 
                          deterministic=True, enable_progress_bar=True, enable_model_summary=True,
                          default_root_dir = path_root, callbacks = [checkpoint_callback],
                          )
    else:
        logging.debug("Training with CPUs")
        trainer = Trainer(accelerator = "cpu", max_epochs = num_epochs, 
                          detect_anomaly = True,
                          num_sanity_val_steps = 0, default_root_dir = path_results, 
                          check_val_every_n_epoch = 1, callbacks = [checkpoint_callback])

    # Train
    trainer.fit(model,data)

    # Test
    #trainer.test(model,data)

    # Dump config
    yaml.dump(params, open(os.path.join(path_root, f'{path_results}/{model_id}/params.yaml'), 'w'))

if __name__ == "__main__":
    params = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    run(params)
