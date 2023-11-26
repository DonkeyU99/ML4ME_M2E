class DataLoader():
    def __init__(self,num_data,img_size,transforms=None):
        self.num_data = num_data
        self.img_size = img_size
        self.transforms = transforms
    
    def get_dataset(self):
        if(self.transforms):
            pass
        pass
    
    def get_datatype(self):
        return self.num_data,self.data_size,self.transforms
    
    def split_train_val(self):
        pass