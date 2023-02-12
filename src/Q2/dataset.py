from torch.utils.data import Dataset
import pandas as pd
from os import listdir
from torchvision.io import read_image

images_path = './VOC_Dataset/images/'
masks_path = './VOC_Dataset/masks/'

class VocDataset(Dataset):
    # Loads the data and populates the X and Y variables
    def __init__(self, transform=None):
        self.transform = transform
        
        # Get the list of image files
        images = listdir(images_path)
        
        # Create a dataframe to load the dataset
        data = []
        for image_name in images:
            image_full_path = images_path + image_name
            mask_full_path = masks_path + image_name[:-3] + 'png'
            image = read_image(image_full_path)
            mask = read_image(mask_full_path)
            data.append([image, mask])
        
        self.df = pd.DataFrame(data, columns=['image', 'mask'])

    # Returns the number of samples in the dataset
    def __len__(self):
        return len(self.df.index)
    
    # Returns a datapoint and label pair at a given index
    def __getitem__(self, idx):
        image = self.df.iloc[idx, 0]
        mask = self.df.iloc[idx, 1]
        
        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, mask
