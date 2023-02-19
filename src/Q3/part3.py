import wandb
wandb.login()

from dataset import SvnhDataset
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF
import multiprocessing
import numpy as np
from copy import deepcopy
import torch

# Set a manual seed for reproducibility
torch.manual_seed(6225)
np.random.seed(6225)

# Set the checkpoint path
checkpoint_path = "./saved_state/yolo_Q3_2.pt"

# Set the pictures directory
pictures_path = "./pictures_part2/"

# Use all cores
torch.set_num_threads(multiprocessing.cpu_count())

# Use GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The configuration for wandb
config = dict(
        train_ratio = 0.7,
        val_ratio = 0.2,
        test_ratio = 0.1,
        dataset = "SVNH",
        adam_beta1 = 0.9, 
        adam_beta2 = 0.999,
        learning_rate = 0.01,
        weight_decay = 0.005,
        batch_size = 32,
        epochs = 10,
        log_interval = 1,
        )

# Custom transform classes
class MyHflip(object):
    """Horizontally flips an image"""

    def __call__(self, img):
        return TF.hflip(img)

class MyVflip(object):
    """Horizontally flips an image"""

    def __call__(self, img):
        return TF.vflip(img)

# Creates the model and the data loaders
def make(config):

    # Create the transform for images
    transform_image_none = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        ])

    transform_blur = transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(3),
        transforms.Resize((32, 32)),
    ])
    
    transform_hflip = transforms.Compose([
        transforms.ToTensor(),
        MyHflip(),
        transforms.Resize((32, 32)),
    ])
    
    transform_vflip = transforms.Compose([
        transforms.ToTensor(),
        MyVflip(),
        transforms.Resize((32, 32)),
    ])

    # Load the data again but this time with transform. Split it into appropriate chunk size
    training_data, validation_data, testing_data = random_split(SvnhDataset(transform=transform_image_none), [config['train_ratio'], config['val_ratio'], config['test_ratio']])
    
    # Make 2 copies of training data for augmentation
    training_data_hflip = deepcopy(training_data)
    training_data_vflip = deepcopy(training_data)
    training_data_blur = deepcopy(training_data)

    # Set the transforms
    training_data_hflip.transform = transform_hflip
    training_data_vflip.transform = transform_vflip
    training_data_blur.transform = transform_blur

    # Concatenate the training dataset with all the three transforms
    training_data_full = ConcatDataset([training_data, training_data_hflip, training_data_vflip, training_data_blur])
    

    # Create data loaders for training, validation and testing sets
    train_loader = DataLoader(training_data_full, shuffle=True, pin_memory=True, batch_size=config['batch_size'])
    val_loader = DataLoader(validation_data, shuffle=True, pin_memory=True, batch_size=config['batch_size'])
    test_loader = DataLoader(testing_data, shuffle=True, pin_memory=True, batch_size=config['batch_size'])

    return train_loader, val_loader, test_loader

def model_pipeline(hyperparameters):

    # Tell wandb to get started
    with wandb.init(project="CV_Assignment01", config=hyperparameters, name="YOLOv5"):
        print("The model will be running on", device, "device")

        # Access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Make the data loaders
        train_loader, val_loader, test_loader = make(config)

if __name__ == "__main__":
    model_pipeline(config)
    wandb.finish()
