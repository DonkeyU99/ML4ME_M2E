# data loader from imagenet
# data augmentation -> resize to (batch, 3, 224, 224) + blur augmentation
import torch
import kornia.augmentation as ka
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

import numpy as np
import matplotlib.pyplot as plt

class My_Data_loader():
    def __init__(self, num_img, img_size, transform = None):
         self.num_img = num_img
         self.img_size = img_size
         self.transform = transform
        
    def get_dataset(self, batch_size, kernel_size):
        if (self.num_img % batch_size) != 0:
            print('number of total data must be divided by batch_size')

        if (self.transform != None):
            tf = transforms.Compose( [transforms.ToTensor(), 
                                      transforms.Resize(self.img_size, antialias = None), 
                                      transforms.Normalize((0.5,0.5,0.5),(1,1,1))])
            
            data = datasets.CIFAR10(root = "datasets/train_data/cifar10", train = True, transform = tf, download = True)
            train_dataloader = DataLoader(data, batch_size = batch_size, shuffle = True)
            dataset = []
            aug = ka.RandomMotionBlur(kernel_size = kernel_size, angle = (-45, 45),direction = (-1,1), keepdim = True)
            cnt = 0
            for img, _ in train_dataloader:
                aug_img = aug(img)
                dataset.append(aug_img)
                cnt += 1
                if cnt == (self.num_img // batch_size):
                    break
             # final dataset(batch_size * self.num_img // batch_size, 3, img_size, img_size)
            dataset_f = torch.cat(dataset, 0)
        
        

        return dataset_f
    


        
    

