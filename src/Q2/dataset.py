from torch.utils.data import Dataset
import pandas as pd
from os import listdir
from torchvision.io import read_image
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

images_path = './VOC_Dataset/images/'
masks_path = './VOC_Dataset/masks/'

class VocDataset(Dataset):
    # Loads the data and populates the X and Y variables
    def __init__(self, transform_image=None, transform_mask=None):
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        
        # Get the list of image files
        images = listdir(images_path)
        
        # Create a dataframe to load the dataset
        data = []
        for image_name in images:
            image_full_path = images_path + image_name
            mask_full_path = masks_path + image_name[:-3] + 'png'
            data.append([image_full_path, mask_full_path])
        
        self.df = pd.DataFrame(data, columns=['image', 'mask'])

    # Returns the number of samples in the dataset
    def __len__(self):
        return len(self.df.index)
    
    # Returns a datapoint and label pair at a given index
    def __getitem__(self, idx):
        image_full_path = self.df.iloc[idx, 0]
        mask_full_path = self.df.iloc[idx, 1]
        
        # Read mask and image
        image = Image.open(image_full_path)
        mask = Image.open(mask_full_path)
        
        mask = np.array(mask.getdata()).reshape(1, mask.size[1], mask.size[0])
        mask[mask == 255] = 0
        
        # Apply transform
        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        mask = mask[0]  # Remove the dummy dimension

        return image, mask
