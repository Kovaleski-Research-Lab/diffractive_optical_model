#--------------------------------
# Import: Python libraries
#--------------------------------

import torch
from loguru import logger

#--------------------------------
# Initialize: Wavefront transform
#--------------------------------

class WavefrontTransform(object):
    def __init__(self, params):
        self.params = params.copy()
        logger.debug("custom_transforms.py - Initializing WavefrontTransform")

        # Set initialization strategy for the wavefront
        self.phase_initialization_strategy = params['phase_initialization_strategy']
        self.bits = int(params.get('bits', 64))
        if self.bits == 64:
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64
        elif self.bits == 128:
            self.real_dtype = torch.float64
            self.complex_dtype = torch.complex128
        else:
            raise ValueError(
                "WavefrontTransform bits must be 64 (complex64) or 128 (complex128)."
            )
        if self.phase_initialization_strategy not in (0, 1):
            raise ValueError("phase_initialization_strategy must be 0 or 1.")

        if self.phase_initialization_strategy == 0:
            logger.debug("custom_transforms.py | WavefrontTransform | Phase Initialization : Phase = 0, Amplitude = Sample")
        else:
            logger.debug("custom_transforms.py | WavefrontTransform | Phase Initialization : Phase = Sample, Amplitude = torch.ones()")

    def __call__(self,sample):
        sample = sample.to(dtype=self.real_dtype)
        if self.phase_initialization_strategy == 0:
            phases = torch.zeros_like(sample)
            amplitude = sample
        else:
            phases = sample
            amplitude = torch.ones_like(sample)

        wavefront = amplitude * torch.exp(1j*phases)
        return wavefront.to(dtype=self.complex_dtype)

#--------------------------------
# Initialize: Normalize transform
#--------------------------------
class Normalize(object):                                                                    
    def __init__(self, params):                                                             
        self.params = params.copy()                                                         
        logger.debug("custom_transforms.py - Initializing Normalize")
                                                                                            
    def __call__(self,sample):                                                              
        sample = sample.to(dtype=torch.get_default_dtype())
        min_val = torch.min(sample)
        sample = sample - min_val                                                           
        max_val = torch.max(sample)
        if not bool(max_val > 0):
            return torch.zeros_like(sample)
        return sample / max_val

#--------------------------------
# Initialize: Threshold transform
#--------------------------------

class Threshold(object):
    def __init__(self, threshold):
        logger.debug("custom_transforms.py - Initializing Threshold")
        self.threshold = threshold
        logger.debug("custom_transforms.py | Threshold | Setting threshold to {}".format(self.threshold))

    def __call__(self, sample):
        return (sample > self.threshold).to(dtype=sample.dtype)
