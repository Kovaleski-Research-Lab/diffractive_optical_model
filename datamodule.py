#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import logging
from typing import Optional
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

import sys
sys.path.append('../')
import custom_transforms as ct

import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
#--------------------------------
# Initialize: MNIST Wavefront
#--------------------------------

class Wavefront_MNIST_DataModule(LightningDataModule):
    def __init__(self, params: dict, transform:str = "") -> None:
        super().__init__() 
        logging.debug("datamodule.py - Initializing Wavefront_MNIST_DataModule")
        self.params = params.copy()
        self.Nx = self.params['Nxp']
        self.Ny = self.params['Nyp']
        self.n_cpus = self.params['n_cpus']
        self.path_data = self.params['path_data']
        self.path_root = self.params['path_root']
        self.path_data = os.path.join(self.path_root,self.path_data)
        logging.debug("datamodule.py - Setting path_data to {}".format(self.path_data))
        self.batch_size = self.params['batch_size']
        self.data_split = self.params['data_split']
        self.initialize_transform()
        self.initialize_cpus(self.n_cpus)

    def initialize_transform(self) -> None:
        resize_row = self.params['resize_row']
        resize_col = self.params['resize_col']

        pad_x = int(torch.div((self.Nx - resize_row), 2, rounding_mode='floor'))
        pad_y = int(torch.div((self.Ny - resize_col), 2, rounding_mode='floor'))

        padding = (pad_y, pad_x, pad_y, pad_x)

        self.transform = transforms.Compose([
                transforms.Resize((resize_row, resize_col), antialias=True), # type: ignore
                transforms.Pad(padding),
                ct.Threshold(0.2),
                ct.WavefrontTransform(self.params['wavefront_transform'])])

    def initialize_cpus(self, n_cpus:int) -> None:
        # Make sure default number of cpus is not more than the system has
        if n_cpus > os.cpu_count(): # type: ignore
            n_cpus = 1
        self.n_cpus = n_cpus 
        logging.debug("Wavefront_MNIST_DataModule | Setting CPUS to {}".format(self.n_cpus))

    def prepare_data(self) -> None:
        MNIST(self.path_data, train=True, download=True)
        MNIST(self.path_data, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        logging.debug("Wavefront_MNIST_DataModule | setup with datasplit = {}".format(self.data_split))
        train_file = f'MNIST/{self.data_split}.split'
        valid_file = 'MNIST/valid.split'

        test_file = 'MNIST/test.split'
        train_data = torch.load(os.path.join(self.path_data, train_file))
        valid_data = torch.load(os.path.join(self.path_data, valid_file))
        test_data = torch.load(os.path.join(self.path_data, test_file))

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
        logging.debug("datamodule.py - Initializing customDataset")
        #self.samples, self.targets = data[0], data[1]
        #shape = data[0].shape
        #self.samples = torch.ones(shape)

        if len(self.samples.shape) < 4:
            self.samples = torch.unsqueeze(self.samples, dim=1)
        if self.samples.shape[1] > 3:
            self.samples = torch.swapaxes(self.samples, 1,-1)

        self.targets = self.samples
        self.transform = transform
        logging.debug("customDataset | Setting transform to {}".format(self.transform))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample,target = self.samples[idx], self.targets[idx]
        sample = self.transform(sample)
        #target = self.transform(target)
        slm_sample = (sample.abs() * 63).to(torch.uint8)
        
        #target = torch.nn.functional.one_hot(torch.tensor(target), num_classes=10)

        return sample, slm_sample, target

#--------------------------------
# Initialize: Select dataset
#--------------------------------

def select_data(params):
    if params['which'] == 'MNIST' :
        return Wavefront_MNIST_DataModule(params) 
    else:
        logging.error("datamodule.py | Dataset {} not implemented!".format(params['which']))
        exit()

#--------------------------------
# Initialize: Testing
#--------------------------------

if __name__=="__main__":
    import yaml
    import torch
    import matplotlib.pyplot as plt
    from pytorch_lightning import seed_everything
    from utils import parameter_manager
    logging.basicConfig(level=logging.DEBUG)
    seed_everything(1337)
    os.environ['SLURM_JOB_ID'] = '0'
    #plt.style.use(['science'])

    #Load config file   
    params = yaml.load(open('../config.yaml'), Loader = yaml.FullLoader).copy()
    params['model_id'] = "test_0"
    #params['path_data'] = '/home/marshall/Documents/research/deep-optics/data/'

    #Parameter manager
    
    dm = select_data(params)
    dm.prepare_data()
    dm.setup(stage="fit")

    #View some of the data

    images,slm_sample, labels = next(iter(dm.train_dataloader()))

    from IPython import embed; embed()
    print(images[0])
    print(dm.train_dataloader().__len__())
    print(images.shape)
    print(labels)

    #fig,ax = plt.subplots(1,3,figsize=(5,5))
    #for i,image in enumerate(images):
    #    ax[i].imshow(image.squeeze().abs())
    #    ax[i].axis('off')

    #plt.show()

