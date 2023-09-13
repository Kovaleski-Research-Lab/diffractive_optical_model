#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import logging
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from core import datamodule, lrn, custom_logger, classifiers, cooperative



def select_model(pm):
    logging.debug("model_loader.py |  Selecting model") 
    model = None
    if pm.lrn and not pm.classifier:
        logging.debug("select_model() | selecting LRN")
        assert model is None
        model = lrn.LRN(pm.params_model_lrn, pm.all_paths,  pm.params_propagator, pm.params_modulator)
        if pm.load_checkpoint_lrn:
            logging.debug("select_model() | Loading lrn checkpoint")
            model.load_from_checkpoint(pm.path_checkpoint_lrn,
                                           params = (pm.params_model_lrn, pm.params_propagator, pm.params_modulator),
                                           strict = True)
        pm.model_name = 'lrn'
        pm.collect_params()

    elif pm.classifier and not pm.lrn:
        logging.debug("select_model() | selecting classifier")
        assert model is None
        model = classifiers.Classifier(pm.params_model_classifier)
        if pm.load_checkpoint_classifier:
            logging.debug("select_model() | Loading classifier checkpoint")
            model.load_from_checkpoint(pm.path_checkpoint_classifier,
                                            params = pm.params_model_classifier,
                                            strict = True)
        pm.model_name = 'classifier'
        pm.collect_params()

    elif pm.classifier and pm.lrn:
        logging.debug("select_model() | selecting cooperative")
        assert model is None
        model = cooperative.CooperativeOptimization(pm.params_model_cooperative, pm.params_model_lrn, pm.params_propagator,
                                           pm.params_modulator, pm.params_model_classifier, pm.all_paths)
        if pm.load_checkpoint_cooperative:
            path_checkpoint = os.path.join(pm.path_root, pm.path_checkpoint_cooperative, pm.model_id)
            logging.debug("select_model() | Loading cooperative checkpoint : {}".format(path_checkpoint))
            checkpoint = torch.load(path_checkpoint)
            model.load_from_checkpoint(pm.path_checkpoint_cooperative, 
                                             params_model_cooperative = pm.params_model_cooperative,
                                             params_model_lrn = pm.params_model_lrn, 
                                             params_propagator = pm.params_propagator, 
                                             params_modulator = pm.params_modulator, 
                                             params_model_classifier = pm.params_model_classifier, 
                                             all_paths = pm.all_paths)
        pm.model_name = 'cooperative'    
        pm.collect_params()

    else:
        logging.error("model_loader.py | Failed to select model")
        exit()
    assert model is not None

    return model
