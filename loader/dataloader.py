# data loader from imagenet
# data augmentation -> resize to (batch, 3, 224, 224) + blur augmentation
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader as datload
import imgaug.augmenters as iaa
from default_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class My_Data_loader(DataLoader):
    def _init_(self, num_data = 10, img_size = 224, transforms=None):
         super().__init__(num_data, img_size)
         self.num_data = num_data
         self.img_size = img_size
         self.transforms = transforms
        
    def get_dataset(self, batch_size, kernel_size):
        
        tf = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor()])
        data = datasets.CIFAR10(root = 'data_sets/cifar10_train/', train = True, download = True, transform = tf)
    
        blur_data = iaa.MotionBlur(k = kernel_size, angle = (-45,45))
        aug_data = datload(blur_data, batch_size= batch_size, shuffle=True)
        return aug_data
    