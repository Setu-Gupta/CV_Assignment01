from torch.utils.data import Dataset
import pandas as pd
from os import listdir
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

images_path = './svhn/images/all/'
labels_path = './svhn/labels/all/'

class SvnhDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        
        # Get the list of image files
        images = listdir(images_path)
        
        # Create a dataframe to load the dataset
        data = []
        for image_name in images:
            image_full_path = images_path + image_name
            label_full_path = labels_path + image_name[:-3] + 'txt'
            data.append([image_full_path, label_full_path])
        
        self.df = pd.DataFrame(data, columns=['image', 'label'])

    # Returns the number of samples in the dataset
    def __len__(self):
        return len(self.df.index)
    
    # Returns a datapoint and label pair at a given index
    def __getitem__(self, idx):
        image_full_path = self.df.iloc[idx, 0]
        label_full_path = self.df.iloc[idx, 1]
        
        # Read the image
        image = Image.open(image_full_path)
        label = []
        with open(label_full_path) as label_file:
            for box in label_file:
                box_label = []
                class_idx, x_center, y_center, width, height = box.split()
                class_idx = int(class_idx)
                x_center = float(x_center)
                y_center = float(y_center)
                width = float(width)
                height = float(height)
                box_label = [class_idx, x_center, y_center, width, height]
                label.append(box_label)

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label
