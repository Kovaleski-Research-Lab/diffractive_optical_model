#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import logging
import torchvision
import pytorch_lightning as pl

#--------------------------------
# Initialize: Propagator
#--------------------------------

class Propagator(pl.LightningModule):
    def __init__(self, params, paths):
        super().__init__()
        logging.debug("propagator.py - Initializing Propagator")

        self.params = params.copy()
        self.paths = paths.copy()

        # Load: Physical parameters
        self.Lxp = torch.tensor(self.params['Lxp'])   
        self.Lyp = torch.tensor(self.params['Lyp'])   
        self.Nxp = torch.tensor(self.params['Nxp'])
        self.Nyp = torch.tensor(self.params['Nyp'])
        self.distance = torch.tensor(params['distance'])
        logging.debug("Propagator | setting distance to {}".format(self.distance))
        self.wavelength = torch.tensor(self.params['wavelength'])
        logging.debug("Propagator | setting wavelength to {}".format(self.wavelength))
        self.wavenumber = 2 * torch.pi / self.wavelength 

        self.delta_x = self.Lxp / self.Nxp
        self.delta_y = self.Lyp / self.Nyp
        logging.debug("Propagator | setting sampling pitch {}x{}".format(self.delta_x, self.delta_y))

        # Initialize: Center crop transform
        self.cc = torchvision.transforms.CenterCrop((int(self.Nxp), int(self.Nyp)))

        #Create: The propagator
        self.asm = None
        self.adaptive = self.params['adaptive']
        logging.debug("Propagator | setting adaptive to {}".format(self.adaptive))

        self.create_propagator()
 
    def create_propagator(self):
        logging.debug("Propagator | creating propagation layer")
        padx = torch.div(self.Nxp, 2, rounding_mode='trunc')
        pady = torch.div(self.Nyp, 2, rounding_mode='trunc')
        self.padding = (pady,pady,padx,padx)    

        if self.check_asm_distance():
            self.asm = True
            self.init_asm_transfer_function()
        else:
            self.asm = False
            self.init_rsc_transfer_function()


    def update_propagator(self):
        logging.debug("Propagator | updating propgator due to specified distance")
        if self.adaptive:
            if self.check_asm_distance():
                self.asm = True
                self.init_asm_transfer_function()
            else:
                self.asm = False
                self.init_rsc_transfer_function()
        else:
            if self.asm:
                self.init_asm_transfer_function()
            else:
                self.init_rsc_transfer_function()

    def check_asm_distance(self):
        logging.debug("Propagator | checking ASM propagation criteria")
        #10.1364/JOSAA.401908 equation 32
        #Checks distance criteria for sampling considerations
        distance_criteria_y = 2 * self.Nyp * (self.delta_y**2) / self.wavelength
        distance_criteria_y *= torch.sqrt(1 - (self.wavelength / (2 * self.Nyp))**2)
       
        distance_criteria_x = 2 * self.Nxp * (self.delta_x**2) / self.wavelength
        distance_criteria_x *= torch.sqrt(1 - (self.wavelength / (2 * self.Nxp))**2)
        
        strict_distance = torch.min(distance_criteria_y, distance_criteria_x) 
        logging.debug("Propagator | maximum propagation distance for asm : {}".format(strict_distance))
    
        return(torch.le(self.distance, strict_distance))
 
    #--------------------------------
    # Initialize: ASM transfer fxn
    #--------------------------------

    def init_asm_transfer_function(self): 
        logging.debug("Propagator | initializing ASM transfer function")
        self.x = torch.linspace(-self.Lxp / 2, self.Lxp / 2, self.Nxp)
        self.y = torch.linspace(-self.Lyp / 2, self.Lyp / 2, self.Nyp)
        self.xx, self.yy = torch.meshgrid(self.x, self.y, indexing='ij')
        
        #Double the number of samples to eliminate asm errors
        self.fx = torch.fft.fftfreq(2*len(self.x), torch.diff(self.x)[0]).to(self.device)
        self.fy = torch.fft.fftfreq(2*len(self.y), torch.diff(self.y)[0]).to(self.device)
        self.fxx, self.fyy = torch.meshgrid(self.fx, self.fy, indexing='ij')

        #Mask out non-propagating waves
        mask = torch.sqrt(self.fxx**2 + self.fyy**2) < (1/self.wavelength)
        self.fxx = mask * self.fxx
        self.fyy = mask * self.fyy
        
        #10.1364/JOSAA.401908 equation 28
        #Also Goodman eq 3-78
        self.fz = torch.sqrt(1 - (self.wavelength*self.fxx)**2 - (self.wavelength*self.fyy)**2).double()
        self.fz *= ((torch.pi * 2)/self.wavelength).to(self.device)

        H = torch.exp(1j * self.distance * self.fz)
        H = torch.fft.fftshift(H)
        H.requrires_grad = False
        self.register_buffer('H', H)  
 
    #--------------------------------
    # Initialize: RSC transfer fxn
    #--------------------------------

    def init_rsc_transfer_function(self):
        logging.debug("Propagator | initializing RSC transfer function")
        #Double the size to eliminate rsc errors.
        self.x = torch.linspace(-self.Lxp , self.Lxp, 2*self.Nxp).to(self.device)
        self.y = torch.linspace(-self.Lyp , self.Lyp, 2*self.Nyp).to(self.device)
        self.xx,self.yy = torch.meshgrid(self.x, self.y, indexing='ij')

        #10.1364/JOSAA.401908 equation 29
        #Also Goodman eq 3-79
        r = torch.sqrt(self.xx**2 + self.yy**2 + self.distance**2).double()
        k = (2 * torch.pi / self.wavelength).double()
        z = self.distance.double()

        h_rsc = torch.exp(1j*k*r) / r
        h_rsc *= ((1/r) - (1j*k))
        h_rsc *= (1/(2*torch.pi)) * (z/r)
        H = torch.fft.fft2(h_rsc)
        mag = H.abs()
        ang = H.angle()
        
        mag = mag / torch.max(mag)
        H = mag * torch.exp(1j*ang)
        H.requrires_grad = False
        self.register_buffer('H', H) 

    #--------------------------------
    # Initialize: Helper to crop
    #--------------------------------

    def center_crop_wavefront(self, wavefront):
        return self.cc(wavefront)
 
    #--------------------------------
    # Initialize: Forward pass
    #--------------------------------

    def forward(self, wavefront, distance = None):

        if distance is not None:
            self.distance = distance
            self.update_propagator()
        if wavefront.shape != self.H.shape:
            wavefront = torch.nn.functional.pad(wavefront,self.padding,mode="constant") 
        # The different methods require a different ordering of the shifts...
        if self.asm:
            A = torch.fft.fft2(wavefront)
            A = torch.fft.fftshift(A, dim=(-1,-2))
            U = A * self.H
            U = torch.fft.ifftshift(U, dim=(-1,-2))
            U = torch.fft.ifft2(U)
        else:
            A = torch.fft.fft2(wavefront)
            U = A * self.H 
            U = torch.fft.ifft2(U)
            U = torch.fft.ifftshift(U, dim=(-1,-2))
        U = self.center_crop_wavefront(U)
        return U

#--------------------------------
# Initialize: Test code
#--------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import yaml
    import math
    #from core import datamodule
    import datamodule
    from utils import parameter_manager

    logging.basicConfig(level=logging.DEBUG)

    params = yaml.load(open("../config.yaml"), Loader=yaml.FullLoader)
    params_prop = params['don']['propagators'][0]

    dm = datamodule.Wavefront_MNIST_DataModule(params)

    #Initialize the data module
    dm.prepare_data()
    dm.setup(stage="fit")
    
    #View some of the data
    image,slm_sample,labels = next(iter(dm.train_dataloader()))
   
    #Propagate an image
    propagator = Propagator(params_prop)
    dp = propagator(image)

    fig,ax = plt.subplots(1,3,figsize=(10,5))
    ax[0].imshow(torch.abs(image.squeeze()))
    ax[1].imshow(torch.abs(dp).squeeze())
    ax[2].imshow(torch.abs(dp).squeeze()**2) 
    plt.show()

