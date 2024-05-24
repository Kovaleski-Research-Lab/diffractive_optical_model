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
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.dirname(__file__))

import modulator
import propagator
import plane
import diffraction_block




#-----------------------------------
# Model: Diffractive optical Network
#-----------------------------------

class DON(LightningModule):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.params = params
        self.training_params = params['don']
        self.select_objective()
        self.create_layers()
        self.learning_rate = self.training_params['learning_rate']
        self.save_hyperparameters()
    
    #--------------------------------
    # Create: Network layers
    #--------------------------------

    def create_layers(self):
        self.layers = torch.nn.ModuleList()
        for block in self.params['diffraction_blocks']:
            block_params = self.params['diffraction_blocks'][block]
            self.layers.append(diffraction_block.DiffractionBlock(block_params))

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
    # Initialize: DON Metrics
    #--------------------------------
    
    def run_don_metrics(self, don_outputs):
        wavefronts = don_outputs[0]
        amplitudes = don_outputs[1] 
        normalized_amplitudes = don_outputs[2]
        images = don_outputs[3]
        normalized_images = don_outputs[4]
        don_target = don_outputs[5]
        mse_vals = mse(normalized_images.detach(), don_target.detach())
        psnr_vals = psnr(normalized_images.detach(), don_target.detach())
        ssim_vals = ssim(normalized_images.detach(), don_target.detach()).detach() #type: ignore
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
    # Create: Optimizer Function
    #--------------------------------
   
    def configure_optimizers(self):
        logger.debug("DON | setting optimizer to ADAM")
        optimizer = torch.optim.Adam(self.layers.parameters(), lr = self.learning_rate)
        return optimizer

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
        samples, slm_sample, targets = batch

        output_wavefronts = self.forward(samples)

        # Get auxiliary outputs
        outputs = self.calculate_auxiliary_outputs(output_wavefronts)

        return outputs, targets
  
    #--------------------------------
    # Create: Training Step
    #--------------------------------
             
    def training_step(self, batch, batch_idx):
        outputs, targets = self.shared_step(batch, batch_idx)

        loss = self.objective(outputs['images'], batch[0].squeeze().abs()**2)

        self.log("train_loss", loss, prog_bar = True) #type: ignore

        ## Detach the tensors in the outputs dictionary 
        #for key in outputs:
        #    outputs[key] = outputs[key].detach()

        return { 'loss' : loss, 'outputs' : outputs, 'target' : targets.detach() }
   
    #--------------------------------
    # Create: Validation Step
    #--------------------------------
                
    def validation_step(self, batch, batch_idx):
        outputs, targets = self.shared_step(batch, batch_idx)

        loss = self.objective(outputs['images'], batch[0].squeeze().abs()**2)

        self.log("val_loss", loss, prog_bar = True) #type: ignore

        # Detach the tensors in the outputs dictionary
        #for key in outputs:
        #    outputs[key] = outputs[key].detach()

        return { 'loss' : loss, 'output' : outputs, 'target' : targets.detach() }
    
    #--------------------------------
    # Create: Post Train Epoch Step
    #--------------------------------
    # I was using this to log the modulator parameters after each epoch       
    #def on_train_epoch_end(self):
    #    modulator = {'phase': self.layers[1].phase.detach(), 'amplitude': self.layers[1].amplitude.detach()}
    #    self.logger.experiment.log_results(modulator, self.current_epoch, "train")


    #--------------------------------
    # Create: Test Step
    #--------------------------------

    def predict_step(self, batch, batch_idx):
        outputs, targets = self.shared_step(batch, batch_idx)
        
        # Detach the tensors in the outputs dictionary
        #for key in outputs:
        #    outputs[key] = outputs[key].detach()

        return { 'output' : outputs, 'target' : targets.detach() }

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    import datamodule
    import yaml
    
    params = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    dm = datamodule.select_data(params)
    #Initialize the data module
    dm.prepare_data()
    dm.setup(stage="fit")
    #View some of the data
    #batch = next(iter(dm.train_dataloader()))

    #wavefront = torch.ones(1,1,1080,1920) * torch.exp(1j * torch.ones(1,1,1080,1920))
    dm = datamodule.Wavefront_MNIST_DataModule(params)

    #Initialize the data module
    dm.prepare_data()
    dm.setup(stage="fit")
    
    #View some of the data
    image,slm_sample,labels = next(iter(dm.train_dataloader()))

    network = DON(params)

    model_params = torch.load('results/my_models/early_testing/epoch=49-step=62500.ckpt')
    state_dict = model_params['state_dict']

    phase = state_dict['layers.1.modulator.phase'].cpu()

    network.layers[1].modulator.phase = torch.nn.Parameter(phase)

    outputs = network.shared_step((image,image,labels), 0)

    from IPython import embed; embed()

    output_wavefronts, amplitudes, normalized_amplitudes, images, normalized_images, target = outputs[0]['output_wavefronts'], outputs[0]['amplitudes'], outputs[0]['normalized_amplitudes'], outputs[0]['images'], outputs[0]['normalized_images'], outputs[1]
     
    fig,ax = plt.subplot_mosaic('aa;bb;cc;dd;ee;ff;gg', figsize=(15,15))

    ax['a'].imshow(output_wavefronts.abs().squeeze().cpu().detach())
    ax['a'].set_title('Wavefront Amplitude')

    ax['b'].imshow(output_wavefronts.angle().squeeze().cpu().detach())
    ax['b'].set_title('Wavefront Phase')

    ax['c'].imshow(amplitudes.squeeze().cpu().detach())
    ax['c'].set_title('Amplitude')
    
    ax['d'].imshow(normalized_amplitudes.squeeze().cpu().detach())
    ax['d'].set_title('Normalized Amplitude')
    
    ax['e'].imshow(images.squeeze().cpu().detach())
    ax['e'].set_title('Intensity')
    
    ax['f'].imshow(normalized_images.squeeze().cpu().detach())
    ax['f'].set_title('Normalized Intensity')
    
    ax['g'].imshow(target.squeeze().cpu().detach())
    ax['g'].set_title('Target')

    plt.tight_layout()
    plt.show()


    mse_vals = mse(normalized_images.squeeze().detach(), image.abs().squeeze().detach()**2)
    print(mse_vals)

