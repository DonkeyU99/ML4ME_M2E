from loader.default_loader import DataLoader
import numpy as np


class Dataloader_3_by_3(DataLoader):
    def __init__(self,num_data=1000,img_size=(3,3),transforms=None):
        super().__init__(num_data,img_size,transforms)
        self.num_data = num_data
        self.img_size = img_size

    def get_dataset(self):
        before = np.random.rand(self.num_data,3,3)
        before_pad = np.pad(before,((0,0),(1,1),(1,1)))
        after = np.zeros((self.num_data,3,3))
        ##blur image using 2x2  avg filter
        for i in range(3):
            for j in range(3):
                after[:,i,j] = np.sum(before_pad[:,i:i+2,j:j+2].reshape(self.num_data,-1),axis=1)
                if(i==1):
                    if(j==1):
                        after[:,i,j] /=4.
                    else:
                        after[:,i,j] /=2.
                else:
                    if(j==1):
                        after[:,i,j] /=2.
                    else:
                        after[:,i,j] /=1.
        
        return before, after