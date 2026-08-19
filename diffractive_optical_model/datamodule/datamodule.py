#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
from loguru import logger
from typing import Optional
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from diffractive_optical_model.datamodule import custom_transforms as ct

import torch
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
        self.valid_rate = self.params.get('valid_rate', 0.1)
        seed_config = self.params.get('seed', [True, 0])
        if isinstance(seed_config, (list, tuple)):
            self.seed = int(seed_config[1]) if len(seed_config) > 1 else 0
        else:
            self.seed = int(seed_config)
        self.initialize_transform()
        self.initialize_cpus(self.n_cpus)

    def initialize_transform(self) -> None:
        resize_row = int(self.params['resize_row'])
        resize_col = int(self.params['resize_col'])
        if resize_row <= 0 or resize_col <= 0:
            raise ValueError("resize_row and resize_col must be strictly positive.")
        if resize_row > self.Nx or resize_col > self.Ny:
            raise ValueError(
                "Resized MNIST shape ({}, {}) exceeds target grid ({}, {}).".format(
                    resize_row, resize_col, self.Nx, self.Ny
                )
            )

        vertical_padding = self.Nx - resize_row
        horizontal_padding = self.Ny - resize_col
        top = vertical_padding // 2
        bottom = vertical_padding - top
        left = horizontal_padding // 2
        right = horizontal_padding - left
        padding = (left, top, right, bottom)

        wavefront_params = dict(self.params['wavefront_transform'])
        wavefront_params.setdefault('bits', self.params.get('bits', 64))

        self.transform = transforms.Compose([
                transforms.Resize((resize_row, resize_col), antialias=True), # type: ignore
                transforms.RandomRotation((90,90)),
                transforms.Pad(padding),
                ct.Normalize({}),
                ct.Threshold(0.2),
                ct.WavefrontTransform(wavefront_params)])

    def initialize_cpus(self, n_cpus:int) -> None:
        # Make sure default number of cpus is not more than the system has
        if isinstance(n_cpus, bool) or int(n_cpus) != n_cpus or n_cpus < 0:
            raise ValueError("n_cpus must be a non-negative integer.")
        available_cpus = os.cpu_count() or 1
        self.n_cpus = min(int(n_cpus), available_cpus)
        logger.debug("Setting CPUS to {}".format(self.n_cpus))

    def prepare_data(self) -> None:
        MNIST(self.path_data, train=True, download=True)
        MNIST(self.path_data, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        logger.debug("Setup()")

        if stage == "fit" or stage is None:
            train_data = customDataset(
                MNIST(self.path_data, train=True, download=False), self.transform
            )
            validation_size = self._validation_size(len(train_data))
            train_size = len(train_data) - validation_size
            generator = torch.Generator().manual_seed(self.seed)
            self.mnist_train, self.mnist_val = random_split(
                train_data, [train_size, validation_size], generator=generator
            )
        if stage in ("test", "predict") or stage is None:
            self.mnist_test = customDataset(
                MNIST(self.path_data, train=False, download=False), self.transform
            )

    def _validation_size(self, dataset_size):
        if isinstance(self.valid_rate, bool):
            raise ValueError("valid_rate must be a fraction in (0, 1) or a sample count.")
        rate = float(self.valid_rate)
        if 0 < rate < 1:
            validation_size = int(round(dataset_size * rate))
        elif rate.is_integer() and 1 <= rate < dataset_size:
            validation_size = int(rate)
        else:
            raise ValueError(
                "valid_rate must be a fraction in (0, 1) or an integer sample count "
                "smaller than the training dataset."
            )
        return max(1, min(validation_size, dataset_size - 1))

    def _loader_kwargs(self):
        return {
            'num_workers': self.n_cpus,
            'persistent_workers': self.n_cpus > 0,
        }

    def train_dataloader(self):
        return DataLoader(self.mnist_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          generator=torch.Generator().manual_seed(self.seed),
                          **self._loader_kwargs())

    def val_dataloader(self):
        return DataLoader(self.mnist_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          **self._loader_kwargs())

    def test_dataloader(self):
        return DataLoader(self.mnist_test,
                          batch_size=1,
                          shuffle=False,
                          **self._loader_kwargs())

    def predict_dataloader(self):
        return DataLoader(self.mnist_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          **self._loader_kwargs())

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
        raise ValueError("Dataset {} not implemented!".format(params['which']))

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

