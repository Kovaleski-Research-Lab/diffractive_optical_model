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
import propagator

class Source(propagator.Propagator):
    def __init__(self, params, paths):
        super().__init__(params, paths)
        logging.debug("source.py | Initializing Source")
        self.params = params
        self.init_wavefront()

    def init_wavefront(self):
        self.wavefront = torch.ones(self.H.shape) * torch.exp(1j*torch.ones(self.H.shape))

    def forward(self, distance = None):
        if distance is not None:
            self.distance = distance
            self.update_propagator()
        # The different methods require a different ordering of the shifts...
        if self.asm:
            A = torch.fft.fft2(self.wavefront)
            A = torch.fft.fftshift(A, dim=(-1,-2))
            U = A * self.H
            U = torch.fft.ifftshift(U, dim=(-1,-2))
            U = torch.fft.ifft2(U)
        else:
            A = torch.fft.fft2(self.wavefront)
            U = A * self.H 
            U = torch.fft.ifft2(U)
            U = torch.fft.ifftshift(U, dim=(-1,-2))
        U = self.center_crop_wavefront(U)
        return U

