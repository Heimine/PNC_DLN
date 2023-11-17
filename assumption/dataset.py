import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision
from PIL import Image

from scipy.stats import ortho_group
# i.i.d. orthogonal dataset
class OrthogonalDataset(Dataset):
    """i.i.d. Orthogonal Dataset."""

    def __init__(self, data, target):
        super(OrthogonalDataset, self).__init__()
        # self.data = ortho_group.rvs(num_samples) # * np.sqrt(num_samples)
        # print(self.data.shape)
        self.data = data 
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
# i.i.d. Gaussian dataset
class GaussianDataset(Dataset):
    """i.i.d. Gaussian Dataset."""

    def __init__(self, data, target):
        super(GaussianDataset, self).__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
# i.i.d. Uniform dataset 
class UniformDataset(Dataset):
    """i.i.d. Gaussian Dataset."""

    def __init__(self, data, target):
        super(UniformDataset, self).__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# MNIST (partial)
class MnistPartial(datasets.MNIST):
    # We only use MNIST dataset with 4 classes
    def __init__(self, root, num_samples, train=True, transform=None, **kwargs):
        super().__init__(root=root,train=train, transform=transform, **kwargs)
        
        print("Using MNIST with only the 0/1/2/3 classes! \n")
        sample_size = int(num_samples // 4)
        self.targets = np.array(self.targets)
        zero_data = self.data[self.targets == 0][:sample_size]
        one_data = self.data[self.targets == 1][:sample_size]
        two_data = self.data[self.targets == 2][:sample_size]
        three_data = self.data[self.targets == 3][:sample_size]

        self.data = np.concatenate([zero_data, one_data, two_data, three_data], 0)
        self.target = np.concatenate([np.zeros(sample_size), np.ones(sample_size), np.ones(sample_size) * 2, np.ones(sample_size) * 3])
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.target[index])
        one_hot_target = np.zeros(4)
        one_hot_target[target] = 1

        if self.transform is not None:
            img = self.transform(img)

        return img.flatten(), one_hot_target# target

def orthogonal_dataset(num_samples, data_dim, num_classes, bs=20):
    # Dataset
    # if data_dim < num_samples:
    #     raise ValueError("d must be greater than N.")
    total_label = torch.zeros(num_samples, num_classes)
    
    if data_dim < num_samples:
        A = torch.randn(num_samples, data_dim)
        total_data = torch.linalg.qr(A)[0].t()[:data_dim]
        # print(total_data.shape)
    else:
        A = torch.randn(data_dim, data_dim)
        Q, R = torch.linalg.qr(A)
        total_data = Q[:, :num_samples]
    total_data = total_data.T
    
    ### Add some perturbation
    n,d = total_data.shape
    delta = torch.randn(n,d)
    delta /= torch.linalg.norm(delta)
    total_data = total_data + delta
    ### Added done
    
    if num_samples % num_classes == 0:
        num_sample_class = int(num_samples // num_classes)
    else:
        raise ValueError(" Please make sure num_samples divides num_classes!")
    for i in range(num_classes):
        total_label[i*num_sample_class:(i+1)*num_sample_class, i] = 1
        
    # dataset and dataloader
    orthogonal_trainset = OrthogonalDataset(total_data, total_label)
    orthogonal_train_loader = torch.utils.data.DataLoader(orthogonal_trainset,
                                                 batch_size=bs)
    
    return orthogonal_trainset, orthogonal_train_loader

def gaussian_dataset(num_samples, data_dim, num_classes, bs=20):
    # Dataset
    total_data = torch.randn(num_samples, data_dim)
    U,S,Vh = torch.linalg.svd(total_data, full_matrices=False)
    S = torch.rand_like(S) + 20
    total_data = U @ torch.diag(S) @ Vh
    if num_samples % num_classes == 0:
        num_sample_class = int(num_samples // num_classes)
    else:
        raise ValueError(" Please make sure num_samples divides num_classes!")
    #total_label = torch.cat([torch.ones(num_sample_class) * cla_idx for cla_idx in range(num_classes)])
    total_label = torch.zeros(num_samples, num_classes)
    for i in range(num_classes):
        total_label[i*num_sample_class:(i+1)*num_sample_class, i] = 1
        
    # dataset and dataloader
    gaussian_trainset = GaussianDataset(total_data, total_label)
    gaussian_train_loader = torch.utils.data.DataLoader(gaussian_trainset,
                                                 batch_size=bs)
    
    return gaussian_trainset, gaussian_train_loader

def uniform_dataset(num_samples, data_dim, num_classes, bs=20):
    # Dataset
    total_data = torch.rand(num_samples, data_dim)
    if num_samples % num_classes == 0:
        num_sample_class = int(num_samples // num_classes)
    else:
        raise ValueError(" Please make sure num_samples divides num_classes!")
    #total_label = torch.cat([torch.ones(num_sample_class) * cla_idx for cla_idx in range(num_classes)])
    total_label = torch.zeros(num_samples, num_classes)
    for i in range(num_classes):
        total_label[i*num_sample_class:(i+1)*num_sample_class, i] = 1
    
    # dataset and dataloader
    uniform_trainset = UniformDataset(total_data, total_label)
    uniform_train_loader = torch.utils.data.DataLoader(uniform_trainset,
                                                 batch_size=bs)
    
    return uniform_trainset, uniform_train_loader

def mnist_4_class(num_samples, bs=100, train=True):
    
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    
    train_dataset = MnistPartial(root="./data",
                                 num_samples=num_samples,
                                 transform=transform_train,
                                 download=True,
                                 train=train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs)

    return train_dataset, train_loader