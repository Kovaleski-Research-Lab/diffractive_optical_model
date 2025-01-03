#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
from loguru import logger
from typing import Optional
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import custom_transforms as ct

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
#--------------------------------
# Initialize: MNIST Wavefront
#--------------------------------

class Wavefront_MNIST_DataModule(LightningDataModule):
    def __init__(self, params: dict, transform:str = "") -> None:
        super().__init__() 
        logger.debug("Initializing Wavefront_MNIST_DataModule")
        self.params = params.copy()
        self.Nx = self.params['Nxp']
        self.Ny = self.params['Nyp']
        self.n_cpus = self.params['n_cpus']
        self.path_data = self.params['paths']['path_data']
        self.path_root = self.params['paths']['path_root']
        self.path_data = os.path.join(self.path_root,self.path_data)
        logger.debug("Setting path_data to {}".format(self.path_data))
        self.batch_size = self.params['batch_size']
        self.initialize_transform()
        self.initialize_cpus(self.n_cpus)

    def initialize_transform(self) -> None:
        resize_row = self.params['resize_row']
        resize_col = self.params['resize_col']

        pad_x = int(torch.div((self.Nx - resize_col), 2, rounding_mode='floor'))
        pad_y = int(torch.div((self.Ny - resize_row), 2, rounding_mode='floor'))

        padding = (pad_y, pad_x, pad_y, pad_x)

        self.transform = transforms.Compose([
                transforms.Resize((resize_row, resize_col), antialias=True), # type: ignore
                transforms.RandomRotation((90,90)),
                transforms.Pad(padding),
                ct.Threshold(0.2),
                ct.WavefrontTransform(self.params['wavefront_transform'])])

    def initialize_cpus(self, n_cpus:int) -> None:
        # Make sure default number of cpus is not more than the system has
        if n_cpus > os.cpu_count(): # type: ignore
            n_cpus = 1
        self.n_cpus = n_cpus 
        logger.debug("Setting CPUS to {}".format(self.n_cpus))

    def prepare_data(self) -> None:
        MNIST(self.path_data, train=True, download=True)
        MNIST(self.path_data, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        logger.debug("Setup()")
        train_data = MNIST(self.path_data, train=True, download=False)
        valid_data = MNIST(self.path_data, train=False, download=False)
        test_data = MNIST(self.path_data, train=False, download=False)

        if stage == "fit" or stage is None:
            self.mnist_train = customDataset(train_data, self.transform)
            self.mnist_val = customDataset(valid_data, self.transform)
        if stage == "test" or stage is None:
            self.mnist_test = customDataset(test_data, self.transform)
        if stage == "predict" or stage is None:
            self.mnist_test = customDataset(test_data, self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test,
                          batch_size=1,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=False)

#--------------------------------
# Initialize: Custom dataset
#--------------------------------

class customDataset(Dataset):
    def __init__(self, data, transform):
        logger.debug("Initializing customDataset")
        self.samples = data.data
        self.targets = data.targets

        self.targets = self.samples
        self.transform = transform
        logger.debug("Setting transform to {}".format(self.transform))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample,target = self.samples[idx], self.targets[idx]
        sample = sample.unsqueeze(0)
        target = target.unsqueeze(0)
        sample = self.transform(sample)
        target = self.transform(target)
        return sample, target

#--------------------------------
# Initialize: Select dataset
#--------------------------------

def select_data(params):
    if params['which'] == 'MNIST' :
        return Wavefront_MNIST_DataModule(params) 
    else:
        logger.error("Dataset {} not implemented!".format(params['which']))
        exit()

#--------------------------------
# Initialize: Testing
#--------------------------------

if __name__=="__main__":
    import yaml
    import torch
    import matplotlib.pyplot as plt
    from pytorch_lightning import seed_everything
    seed_everything(1337)
    os.environ['SLURM_JOB_ID'] = '0'
    #plt.style.use(['science'])


    #Load config file   
    params = yaml.load(open('../../config.yaml'), Loader = yaml.FullLoader).copy()
    params['batch_size'] = 3
    params['model_id'] = "test_0"
    params['paths']['path_root'] = '../../'
    
    dm = select_data(params)
    dm.prepare_data()
    dm.setup(stage="fit")

    #View some of the data
    images, labels = next(iter(dm.train_dataloader()))

    from IPython import embed; embed()

    print(images[0])
    print(dm.train_dataloader().__len__())
    print(images.shape)
    print(labels)

    fig,ax = plt.subplots(1,3,figsize=(5,5))
    for i,image in enumerate(images):
        ax[i].imshow(image.squeeze().abs())
        ax[i].axis('off')

    plt.show()

