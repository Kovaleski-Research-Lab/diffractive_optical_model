import numpy as np
import torch

from torchvision import transforms

import torchvision.datasets as datasets
import matplotlib.pyplot as plt

import os
import sys


def gen_train_splits():
    dataset = datasets.MNIST('../data/', train=True, download=True, transform=transforms.ToTensor())

    x = np.array(dataset.data)
    y = np.array(dataset.targets, dtype=np.int64)
    
    num_samples = x.shape[0]
    
    np.random.seed(123)

    indices = np.random.permutation(num_samples)

    x = x[indices]
    y = y[indices]

    s = []
    t = []

    class_samples = {}
    for i in range(0,10):
        class_samples[i] = x[np.where(y == i)]


    single_sample_0, single_target_0 = torch.tensor(class_samples[1][1]), torch.tensor(1)
    single_sample_1, single_target_1 = torch.tensor(class_samples[8][0]), torch.tensor(8)

    one_per_class_samples = []
    one_per_class_targets = []
    for i in class_samples:
        one_per_class_samples.append(class_samples[i][0])
        one_per_class_targets.append(i)
    one_per_class_samples = torch.tensor(np.array(one_per_class_samples))
    one_per_class_targets = torch.tensor(np.array(one_per_class_targets))
    
    ten_per_class_samples = []
    ten_per_class_targets = []
    for i in class_samples:
        ten_per_class_samples.append(class_samples[i][:10])
        ten_per_class_targets.append(np.repeat(i,10))
    ten_per_class_samples = torch.tensor(np.array(ten_per_class_samples))
    ten_per_class_targets = torch.tensor(np.array(ten_per_class_targets))

    hundred_per_class_samples = []
    hundred_per_class_targets = []
    for i in class_samples:
        hundred_per_class_samples.append(class_samples[i][:100])
        hundred_per_class_targets.append(np.repeat(i,100))
    hundred_per_class_samples = torch.tensor(np.array(hundred_per_class_samples))
    hundred_per_class_targets = torch.tensor(np.array(hundred_per_class_targets))


    thousand_per_class_samples = []
    thousand_per_class_targets = []
    for i in class_samples:
        thousand_per_class_samples.append(class_samples[i][:1000])
        thousand_per_class_targets.append(np.repeat(i,1000))
    thousand_per_class_samples = torch.tensor(np.array(thousand_per_class_samples))
    thousand_per_class_targets = torch.tensor(np.array(thousand_per_class_targets))




    #Now to repeat / reshape them to create 10k samples each
    single_sample_0 = torch.unsqueeze(single_sample_0, dim=0)
    single_sample_0 = single_sample_0.repeat(10000,1,1)
    single_target_0 = single_target_0.repeat(10000)
    print(single_sample_0.shape) 
    print(single_target_0.shape) 

    single_sample_1 = torch.unsqueeze(single_sample_1, dim=0)
    single_sample_1 = single_sample_1.repeat(10000,1,1)
    single_target_1 = single_target_1.repeat(10000)
    print(single_sample_1.shape) 
    print(single_target_1.shape) 
   

    one_per_class_samples = torch.reshape(one_per_class_samples, (10,28,28))
    one_per_class_targets = torch.reshape(one_per_class_targets, (10,))
    one_per_class_samples = one_per_class_samples.repeat(1000,1,1)
    one_per_class_targets = one_per_class_targets.repeat(1000)
    print(one_per_class_samples.shape)
    print(one_per_class_targets.shape)
    

    ten_per_class_samples = torch.reshape(ten_per_class_samples, (100,28,28))
    ten_per_class_targets = torch.reshape(ten_per_class_targets, (100,))
    ten_per_class_samples = ten_per_class_samples.repeat(100,1,1)
    ten_per_class_targets = ten_per_class_targets.repeat(100)
    print(ten_per_class_samples.shape)
    print(ten_per_class_targets.shape)

    hundred_per_class_samples = torch.reshape(hundred_per_class_samples, (1000,28,28))
    hundred_per_class_targets = torch.reshape(hundred_per_class_targets, (1000,))
    hundred_per_class_samples = hundred_per_class_samples.repeat(10,1,1)
    hundred_per_class_targets = hundred_per_class_targets.repeat(10)
    print(hundred_per_class_samples.shape)
    print(hundred_per_class_targets.shape)

    thousand_per_class_samples = torch.reshape(thousand_per_class_samples, (10000,28,28))
    thousand_per_class_targets = torch.reshape(thousand_per_class_targets, (10000,))
    print(thousand_per_class_samples.shape)
    print(thousand_per_class_targets.shape)


    all_ones_samples = class_samples[1]
    hundred_ones_samples = torch.tensor(all_ones_samples[0:100])
    hundred_ones_targets = torch.ones(100)

    hundred_ones_samples = torch.reshape(hundred_ones_samples, (100,28,28))
    hundred_ones_samples = hundred_ones_samples.repeat(100,1,1)
    hundred_ones_targets = torch.reshape(hundred_ones_targets, (100,))
    hundred_ones_targets = hundred_ones_targets.repeat(100)
    print(hundred_ones_samples.shape)
    print(hundred_ones_targets.shape)

    ten_ones_samples = torch.tensor(all_ones_samples[0:10])
    ten_ones_targets = torch.ones(10)
    ten_ones_samples = torch.reshape(ten_ones_samples, (10,28,28))
    ten_ones_samples = ten_ones_samples.repeat(1000,1,1)
    ten_ones_targets = torch.reshape(ten_ones_targets, (10,))
    ten_ones_targets = ten_ones_targets.repeat(1000)
    print(ten_ones_samples.shape)
    print(ten_ones_targets.shape)

    all_eight_samples = class_samples[8]
    hundred_eights_samples = torch.tensor(all_eight_samples[0:100])
    hundred_eights_targets = torch.ones(100)*8 
    hundred_eights_samples = torch.reshape(hundred_eights_samples, (100,28,28))
    hundred_eights_samples = hundred_eights_samples.repeat(100,1,1)
    hundred_eights_targets = torch.reshape(hundred_eights_targets, (100,))
    hundred_eights_targets = hundred_eights_targets.repeat(100)
    print(hundred_eights_samples.shape)
    print(hundred_eights_targets.shape)

    ten_eights_samples = torch.tensor(all_eight_samples[0:10])
    ten_eights_targets = torch.ones(10)*8 
    ten_eights_samples = torch.reshape(ten_eights_samples, (10,28,28))
    ten_eights_samples = ten_eights_samples.repeat(1000,1,1)
    ten_eights_targets = torch.reshape(ten_eights_targets, (10,))
    ten_eights_targets = ten_eights_targets.repeat(1000)
    print(ten_eights_samples.shape)
    print(ten_eights_targets.shape)

    #Save them to disk
    torch.save((single_sample_0, single_target_0), '../data/MNIST/mnist_single0.split')
    torch.save((single_sample_1, single_target_1), '../data/MNIST/mnist_single1.split')
    torch.save((one_per_class_samples, one_per_class_targets), '../data/MNIST/mnist_1perClass.split')
    torch.save((ten_per_class_samples, ten_per_class_targets), '../data/MNIST/mnist_10perClass.split')
    torch.save((hundred_per_class_samples, hundred_per_class_targets), '../data/MNIST/mnist_100perClass.split')   
    torch.save((thousand_per_class_samples, thousand_per_class_targets), '../data/MNIST/mnist_1000perClass.split')   

    torch.save((hundred_ones_samples, hundred_ones_targets), '../data/MNIST/mnist_100_1.split')   
    torch.save((hundred_eights_samples, hundred_eights_targets), '../data/MNIST/mnist_100_8.split')   

    torch.save((ten_ones_samples, ten_ones_targets), '../data/MNIST/mnist_10_1.split')   
    torch.save((ten_eights_samples, ten_eights_targets), '../data/MNIST/mnist_10_8.split')   



def gen_test_val_splits():

    np.random.seed(123)

    dataset = datasets.MNIST('../data/', train=False, download=True, transform=transforms.ToTensor())
    x = np.array(dataset.data)
    y = np.array(dataset.targets, dtype=np.int64)
    num_samples = x.shape[0]
    
    indices = np.random.permutation(num_samples)
    x = x[indices]
    y = y[indices]
    s = []
    t = []

    class_samples = {}
    for i in range(0,10):
        class_samples[i] = x[np.where(y == i)]


    val_samples = []
    val_targets = []
    for i in class_samples:
        val_samples.append(class_samples[i][:100])
        val_targets.append(np.repeat(i,100))

    val_samples = torch.from_numpy(np.asarray(val_samples))
    val_targets = torch.from_numpy(np.asarray(val_targets))

    val_samples = torch.reshape(val_samples, (1000,28,28))
    val_targets = torch.reshape(val_targets, (1000,))

    test_samples = []
    test_targets = []
    for i in class_samples:
        test_samples.append(class_samples[i][100:200])
        test_targets.append(np.repeat(i,100))

    test_samples = torch.from_numpy(np.asarray(test_samples))
    test_targets = torch.from_numpy(np.asarray(test_targets))

    test_samples = torch.reshape(test_samples, (1000,28,28))
    test_targets = torch.reshape(test_targets, (1000,))

    torch.save((val_samples, val_targets), '../data/MNIST/valid.split')
    torch.save((test_samples, test_targets), '../data/MNIST/test.split')
    



if __name__ == "__main__":

    gen_train_splits()
    gen_test_val_splits()

