from torch.utils.data import Dataset
import scipy.io

class SvnhDataset(Dataset):
    X = None    # Images
    Y = None    # Labels

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
        label = self.Y[idx].item()
        return image, label
