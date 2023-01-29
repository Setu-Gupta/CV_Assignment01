from torch.utils.data import Dataset
import scipy.io
import numpy as np

class SvnhDataset(Dataset):
    # Loads the data and populates the X and Y variables
    def __init__(self):
        mat =  scipy.io.loadmat('./data/train_32x32.mat')
        self.X = mat['X']
        self.Y = mat['y']

    # Returns the number of samples in the dataset
    def __len__(self):
        return self.Y.shape[0]
    
    # Returns a datapoint and label pair at a given index
    def __getitem__(self, idx):
        image = self.X[:,:,:,idx]
        image = image.astype('float32') # Covert to float32 for pytorch
        image = np.transpose(image, axes=[2, 0, 1])  # Transpose the image to conform with pytorch's input 
        
        label = self.Y[idx].item()
        
        return image, label
