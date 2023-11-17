import torch
import numpy as np
import scipy
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import os
from PIL import Image
import pickle

# Cifar-10 partial
class Cifar10Partial(CIFAR10):
    def __init__(self, root, num_classes, sample_size, train=True, transform=None, download=True, **kwargs):
        super().__init__(root=root,train=train, transform=transform, download=True, **kwargs)
        
        self.num_classes = num_classes
        if num_classes == 0:
            self.data = []
            self.targets = []
        else:
            print(f"Using Cifar10 with only {num_classes} classes! \n")
            self.targets = np.array(self.targets)
            all_data = []
            all_label = []
            for cla in range(num_classes):
                all_data.append(self.data[self.targets == cla][:sample_size])
                class_size = len(self.data[self.targets == cla][:sample_size])
                all_label.append(np.ones(class_size) * cla)

            self.data = np.concatenate(all_data, 0)
            self.targets = np.concatenate(all_label)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        one_hot_target = np.zeros(self.num_classes)
        one_hot_target[target] = 1
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img.flatten(), one_hot_target
    
def get_data(name, data_dir, sample_size, batch_size, num_classes, shuffle=True):
    """
    args:
    @ name: Cifar-10
    @ data_dir: where dataset are stored or to be stored
    @ batch_size: training and testing batch size
    @ num_classes: how many classes to use for the dataset
    """
    
    # Transform 
    transform_cifar10 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    
    if name == "cifar10":
        data_set = Cifar10Partial
        transform_data = transform_cifar10
        
        trainset = data_set(data_dir, num_classes, sample_size,
                        train=True, transform=transform_data)
        testset = data_set(data_dir, num_classes, sample_size,
                            train=False, transform=transform_data)
    
    # Dataloader
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return trainloader, testloader

def get_data_set(name, data_dir, sample_size, num_classes, shuffle=True):
    """
    args:
    @ name: MNIST or Cifar-10
    @ data_dir: where dataset are stored or to be stored
    @ batch_size: training and testing batch size
    @ num_classes: how many classes to use for the dataset
    """
    
    # Transform 
    transform_cifar10 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    
    if name == "cifar10":
        data_set = Cifar10Partial
        transform_data = transform_cifar10
        
        trainset = data_set(data_dir, num_classes, sample_size,
                        train=True, transform=transform_data)
        testset = data_set(data_dir, num_classes, sample_size,
                            train=False, transform=transform_data)
    
    return trainset, testset