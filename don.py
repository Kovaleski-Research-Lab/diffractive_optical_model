#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import logging
import torchmetrics
from pytorch_lightning import LightningModule

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
import sys
sys.path.append('../')

from core.modulator import Modulator, Lens
from core.propagator import Propagator
from core.datamodule import * 
from torchmetrics.functional import mean_squared_error as mse
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

#-----------------------------------
# Model: Diffractive optical Network
#-----------------------------------

class DON(LightningModule):
    def __init__(self, params:dict) -> None:

        super().__init__()
        logging.debug("don.py - Initializing DON")
        # Copy parameters
        self.params = params.copy()
        self.checkpoint_path = self.params['path_checkpoint_don']

        # Create model layers
        self.layers = torch.nn.ModuleList()
        self.create_layers()
        logging.debug("DON | layers initialized to : {}".format(self.layers))
        
        # Select objective function
        self.select_objective()
    
        self.don_metrics = []

        # Training parameters
        self.learning_rate = self.params['learning_rate_don']
        self.save_hyperparameters()
    
    #--------------------------------
    # Create: Network layers
    #--------------------------------

    def create_layers(self):
        logging.debug("DON | creating layers")
        don_params = params['don']
        num_propagators = len(don_params['propagators'])
        num_modulators = len(don_params['modulators'])
        assert num_propagators -1 == num_modulators
        
        print(num_propagators)
        print(num_modulators)
        for i in range(0, num_propagators, 2):
            self.layers.append(Propagator(don_params['propagators'][i]))
            self.layers.append(Modulator(don_params['modulators'][i]))
            self.layers.append(Propagator(don_params['propagators'][i+1]))

    #--------------------------------
    # Select: Objective Function
    #--------------------------------
   
    def select_objective(self):
        objective_function = self.params['objective_function_don']
        if objective_function == "mse":
            self.similarity_metric = False
            self.objective_function = torchmetrics.functional.mean_squared_error
            logging.debug("DON | setting objective function to {}".format(self.objective_function))
        elif objective_function == "psnr":
            self.similarity_metric = True
            self.objective_function = torchmetrics.functional.peak_signal_noise_ratio
            logging.debug("DON | setting objective function to {}".format(self.objective_function))
        elif objective_function == "ssim":
            self.similarity_metric = True
            self.objective_function = torchmetrics.functional.structural_similarity_index_measure
            logging.debug("DON | setting objective function to {}".format(self.objective_function))
        else:
            logging.error("Objective function : {} not supported".format(self.params['objective_function']))
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
        ssim_vals = ssim(normalized_images.detach(), don_target.detach()).detach()

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
        logging.debug("DON | setting optimizer to ADAM")
        optimizer = torch.optim.Adam(self.layers.parameters(), lr = self.learning_rate)
        return optimizer

    #--------------------------------
    # Create: Forward Pass
    #--------------------------------
   
    def forward(self, wavefront):
        x = wavefront
        for i,l in enumerate(self.layers):
            x = l(x)
        x = torch.rot90(x,2,[-2,-1])
        return x
 
    #--------------------------------
    # Create: Shared Step Train/Valid
    #--------------------------------
      
    def shared_step(self, batch, batch_idx):
        sample,target = batch
        output_wavefronts = self(sample)

        #Get the amplitudes
        amplitudes = output_wavefronts.abs()
        
        #Normalized amplitudes
        normalized_amplitudes = torch.cat([amplitude / torch.max(amplitude) for amplitude in amplitudes])
        if len(normalized_amplitudes.shape) < 4:
            normalized_amplitudes = torch.unsqueeze(normalized_amplitudes, dim=1)

        # Calc : Images (intensity of wavefront)
        images = amplitudes**2 

        # Calc : Normalized images
        normalized_images = torch.cat([image / torch.max(image) for image in images])
        if len(normalized_images.shape) < 4:
            normalized_images = torch.unsqueeze(normalized_images, dim=1)

        target = sample.double()

        return output_wavefronts, amplitudes, normalized_amplitudes, images, normalized_images, target
  
    #--------------------------------
    # Create: Training Step
    #--------------------------------
             
    def training_step(self, batch, batch_idx):
        output, amplitudes, normalized_amplitudes, images, normalized_images, target = self.shared_step(batch, batch_idx)

        loss = self.objective(images,target)

        self.log("train_loss", loss, prog_bar = True)
        return { 'loss' : loss, 'output' : output.detach(), 'target' : target.detach() }
   
    #--------------------------------
    # Create: Validation Step
    #--------------------------------
                
    def validation_step(self, batch, batch_idx):
        output, amplitudes, normalized_amplitudes, images, normalized_images, target = self.shared_step(batch, batch_idx)
        
        loss = self.objective(images, target)

        self.log("val_loss", loss, prog_bar = True)
        return { 'loss' : loss, 'output' : output, 'target' : target }
    
    #--------------------------------
    # Create: Post Train Epoch Step
    #--------------------------------
           
    #def on_train_epoch_end(self):
    #    modulator = {'phase': self.layers[1].phase.detach(), 'amplitude': self.layers[1].amplitude.detach()}
    #    self.logger.experiment.log_results(modulator, self.current_epoch, "train")

    def predict_step(self, batch, batch_idx):
       sample, target = batch
       ouput, images, normalized_images, target = self.shared_step(batch, batch_idx)
       return {'output' : output,'images': images, 'normalized_images': normalized_images, 
                'target': target, 'sample' : sample}

    #--------------------------------
    # Create: Test Step
    #--------------------------------
    
    def test_step(self, batch, batch_idx):
        sample, target = batch
        don_output = self.shared_step(batch, batch_idx)
        self.don_metrics.append(self.run_don_metrics(don_output))

        return

    def on_test_end(self):
        #This is a little hacky, but it gets the model id...
        model_id = self.all_paths['path_results_don'].split('/')[-2]
        filename = 'testResults_{}.pt'.format(model_id)
        save_dict = {'don' : self.don_metrics}
        torch.save(save_dict, os.path.join(self.all_paths['path_root'],self.all_paths['path_results_don'],filename))
        return




if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    import yaml
    import math
    #from core import datamodule
    import datamodule
    from utils import parameter_manager

    logging.basicConfig(level=logging.DEBUG)
    
    params = yaml.load(open("../config.yaml"), Loader=yaml.FullLoader)

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

    outputs = network.shared_step((image,labels), 0)

    output_wavefronts, amplitudes, normalized_amplitudes, images, normalized_images, target = outputs

     
    fig,ax = plt.subplot_mosaic('aa;bb;cc;dd;ee;ff;gg', figsize=(15,15))

    ax['a'].imshow(output_wavefronts.abs().squeeze().cpu().detach())
    ax['b'].imshow(output_wavefronts.angle().squeeze().cpu().detach())
    ax['c'].imshow(amplitudes.squeeze().cpu().detach())
    ax['d'].imshow(normalized_amplitudes.squeeze().cpu().detach())
    ax['e'].imshow(images.squeeze().cpu().detach())
    ax['f'].imshow(normalized_images.squeeze().cpu().detach())
    ax['g'].imshow(target.squeeze().cpu().detach())

    plt.tight_layout()
    plt.show()


