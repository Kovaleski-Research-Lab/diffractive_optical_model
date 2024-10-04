#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import sys
import torch
from loguru import logger
import torchmetrics
from IPython import embed

#--------------------------------
# Import: PyTorch Libraries
#--------------------------------

from pytorch_lightning import LightningModule
from torchmetrics.functional import mean_squared_error as mse
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
from diffractive_optical_model.diffraction_block.diffraction_block import DiffractionBlock


#-----------------------------------
# Model: Diffractive optical model
#-----------------------------------

class DOM(LightningModule):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.params = params
        self.training_params = params['dom_training']
        self.select_objective()
        self.create_layers()
        self.learning_rate = self.training_params['learning_rate']
        self.save_hyperparameters()

    #--------------------------------
    # Create: Optimizer Function
    #--------------------------------
   
    def configure_optimizers(self):
        logger.debug("DON | setting optimizer to ADAM")
        optimizer = torch.optim.Adam(self.layers.parameters(), lr = self.learning_rate)
        return optimizer

    #--------------------------------
    # Select: Objective Function
    #--------------------------------
   
    def select_objective(self):
        objective_function = self.training_params['objective_function']
        if objective_function == "mse":
            self.similarity_metric = False
            self.objective_function = torchmetrics.functional.mean_squared_error
            logger.debug("DON | setting objective function to {}".format(objective_function))
        elif objective_function == "psnr":
            self.similarity_metric = True
            self.objective_function = torchmetrics.functional.peak_signal_noise_ratio
            logger.debug("DON | setting objective function to {}".format(objective_function))
        elif objective_function == "ssim":
            self.similarity_metric = True
            self.objective_function = torchmetrics.functional.structural_similarity_index_measure
            logger.debug("DON | setting objective function to {}".format(objective_function))
        else:
            logger.error("Objective function : {} not supported".format(self.training_params['objective_function']))
            exit()

    #--------------------------------
    # Create: Network layers
    #--------------------------------

    def create_layers(self):
        self.layers = torch.nn.ModuleList()
        for block in self.params['diffraction_blocks']:
            block_params = self.params['diffraction_blocks'][block]
            self.layers.append(DiffractionBlock(block_params))

    #--------------------------------
    # Initialize: DOM Metrics
    #--------------------------------
    
    def run_dom_metrics(self, dom_outputs, targets):
        wavefronts = don_outputs['output_wavefronts']
        amplitudes = don_outputs['amplitudes']
        normalized_amplitudes = don_outputs['normalized_amplitudes']
        images = don_outputs['images']
        normalized_images = don_outputs['normalized_images']

        mse_vals = mse(images.detach(), targets.detach())
        psnr_vals = psnr(images.detach(), targets.detach())
        ssim_vals = ssim(images.detach(), targets.detach()).detach() #type: ignore
        return {'mse' : mse_vals.cpu(), 'psnr' : psnr_vals.cpu(), 'ssim' : ssim_vals.cpu()}

    #--------------------------------
    # Initialize: Objective Function
    #--------------------------------
 
    def objective(self, output, target):
        if self.similarity_metric:
            return 1 / (1 + self.objective_function(preds = output, target = target))
        else:
            return self.objective_function(preds = output, target = target)

    #--------------------------------
    # Create: Auxiliary Outputs
    #--------------------------------

    def calculate_auxiliary_outputs(self, output_wavefronts) -> dict:
        amplitudes = output_wavefronts.abs()
        normalized_amplitudes = amplitudes / torch.max(amplitudes)
        images = (amplitudes**2).squeeze()
        normalized_images = images / torch.max(images)
        return {'output_wavefronts' : output_wavefronts, 'amplitudes' : amplitudes,
                'normalized_amplitudes' : normalized_amplitudes, 'images' : images, 
                'normalized_images' : normalized_images}

    #--------------------------------
    # Create: Forward Pass
    #--------------------------------
   
    def forward(self, u:torch.Tensor):
        # Iterate through the layers
        for i,layer in enumerate(self.layers):
            u = layer(u)
        u = torch.rot90(u, 2, [-2,-1])
        return u
 
    #--------------------------------
    # Create: Shared Step Train/Valid
    #--------------------------------
      
    def shared_step(self, batch, batch_idx):
        samples, targets = batch
        output_wavefronts = self.forward(samples)
        # Get auxiliary outputs
        outputs = self.calculate_auxiliary_outputs(output_wavefronts)
        return outputs, targets
  
    #--------------------------------
    # Create: Training Step
    #--------------------------------
             
    def training_step(self, batch, batch_idx):
        outputs, targets = self.shared_step(batch, batch_idx)
        loss = self.objective(outputs['images'], batch[1].squeeze().abs()**2)
        self.log("train_loss", loss, prog_bar = True) #type: ignore
        return { 'loss' : loss, 'outputs' : outputs, 'target' : targets.detach() }
   
    #--------------------------------
    # Create: Validation Step
    #--------------------------------
                
    def validation_step(self, batch, batch_idx):
        outputs, targets = self.shared_step(batch, batch_idx)
        loss = self.objective(outputs['images'], batch[1].squeeze().abs()**2)
        self.log("val_loss", loss, prog_bar = True) #type: ignore
        return { 'loss' : loss, 'output' : outputs, 'target' : targets.detach() }

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    from datamodule.datamodule import select_data
    import yaml
    
    params = yaml.load(open("../config.yaml"), Loader=yaml.FullLoader)
    paths = params['paths']
    path_root = '../'
    paths['path_root'] = path_root
    params['paths'] = paths 
    dm = select_data(params)
    #Initialize the data module
    dm.prepare_data()
    dm.setup(stage="fit")

    # Get some data
    batch = next(iter(dm.train_dataloader()))

    model = DOM(params)
    

    samples, targets = batch
    outputs = model(samples)

    samples = torch.rot90(samples, 1, [-2,-1])
    outputs = torch.rot90(outputs, 1, [-2,-1])

    fig, axs = plt.subplots(1,3)

    sample_image = samples[0].squeeze().abs().numpy()
    output_image = outputs[0].squeeze().abs().numpy()**2
    output_image = output_image / np.max(output_image)
    difference_image = np.abs(sample_image - output_image)

    axs[0].imshow(sample_image)
    axs[1].imshow(output_image)
    axs[2].imshow(difference_image)

    plt.show()

