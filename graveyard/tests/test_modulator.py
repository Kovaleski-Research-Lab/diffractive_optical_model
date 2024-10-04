import sys
import torch
from loguru import logger
import pytorch_lightning as pl
import numpy as np
import unittest
import os
import matplotlib.pyplot as plt
import itertools

sys.path.append('../')
from modulator import *
from plane import Plane


def test():
    modulator_factory = ModulatorFactory()
    logger.info("Running ModulatorFactory tests")
    amplitude_initializations = ['uniform', 'random', 'custom', 'gibberish']
    phase_initializations = ['uniform', 'random', 'custom', 'gibberish']
    types = ['phase_only', 'amplitude_only', 'complex', 'gibberish']
    Nx = 1920
    Ny = 1080
    wavelength = torch.tensor(1.55e-6)
    focal_length = torch.tensor(10.e-2)

    plane_params = { 
            "name" : "test_plane",
            "center" : (0,0,0),
            "size" : (8.96e-3, 8.96e-3),
            "normal_vector" : (0,0,1),
            "Nx" : Nx,
            "Ny" : Ny,
        }

    #Need to create the geometry for the modulators
    test_plane = Plane(plane_params)
    test_cases = itertools.product(amplitude_initializations, phase_initializations, types)
    for t in test_cases:
        phase_pattern = lensPhase(test_plane, wavelength, focal_length) if t[1] == 'custom' else None
        amplitude_pattern = lensPhase(test_plane, wavelength, focal_length) if t[0] == 'custom' else None
        mod_params = {
            "amplitude_init": t[0],
            "phase_init" : t[1],
            "type" : t[2],
            "phase_pattern" : phase_pattern,
            "amplitude_pattern" : amplitude_pattern
        }
        try:
            mod = modulator_factory.create_modulator(test_plane, mod_params)
        except Exception as e:
            if 'gibberish' in t:
                logger.success("Test succeeded: {}".format(t))
                logger.success("Exception caught: {}".format(e))
            else:
                logger.error("Test failed: {}".format(t))
                logger.error("Exception caught: {}".format(e))
        else:
            logger.success("Test succeeded: {}".format(t))



if __name__ == '__main__':
    test()
