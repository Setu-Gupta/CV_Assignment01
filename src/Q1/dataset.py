from torch.utils.data import Dataset
import scipy.io
import numpy as np

class SvnhDataset(Dataset):
    # Loads the data and populates the X and Y variables
    def __init__(self, transform=None):
        mat =  scipy.io.loadmat('./data/train_32x32.mat')
        self.X = mat['X']
        self.Y = mat['y']
        self.transform = transform

    # Returns the number of samples in the dataset
    def __len__(self):
        return self.Y.shape[0]
    
    # Returns a datapoint and label pair at a given index
    def __getitem__(self, idx):
        image = self.X[:,:,:,idx]
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Label 10 is for zero
        label = self.Y[idx].item()  
        if(label == 10):
            label = 0
        
        return image, label
