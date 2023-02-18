import wandb
wandb.login()

from dataset import SvnhDataset
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from torchvision import transforms
import multiprocessing
import numpy as np
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

# Creates the model and the data loaders
def make(config):

    # Create the transform for images
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        ])

    # Load the data again but this time with transform. Split it into appropriate chunk size
    training_data, validation_data, testing_data = random_split(SvnhDataset(transform=transform_image), [config['train_ratio'], config['val_ratio'], config['test_ratio']])

    # Create data loaders for training, validation and testing sets
    train_loader = DataLoader(training_data, shuffle=True, pin_memory=True, batch_size=config['batch_size'])
    val_loader = DataLoader(validation_data, shuffle=True, pin_memory=True, batch_size=config['batch_size'])
    test_loader = DataLoader(testing_data, shuffle=True, pin_memory=True, batch_size=config['batch_size'])

    # Visualize the data distribution
    # Get the target labels
    with torch.no_grad():   # Ensure that no intermidiate results are stored on the GPU
        train_targets = torch.zeros((10), dtype=int)
        val_targets = torch.zeros((10), dtype=int) 
        test_targets = torch.zeros((10), dtype=int)
        
        for _, labels in train_loader:
            for label in labels:
                for class_idx, box in enumerate(label):
                    if(box.sum().item() != 0):
                        train_targets[class_idx] += 1

        for _, labels in val_loader:
            for label in labels:
                for class_idx, box in enumerate(label):
                    if(box.sum().item() != 0):
                        val_targets[class_idx] += 1

        for _, labels in test_loader:
            for label in labels:
                for class_idx, box in enumerate(label):
                    if(box.sum().item() != 0):
                        test_targets[class_idx] += 1

        figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        xlabels = [str(x) for x in range(10)]

        axes[0].bar(xlabels, train_targets)
        axes[0].set_title("Class-Wise Data Distribution [Train]")
        axes[0].set_xlabel("Target Labels")
        axes[0].set_ylabel("Count")

        axes[1].bar(xlabels, val_targets)
        axes[1].set_title("Class-Wise Data Distribution [Validation]")
        axes[1].set_xlabel("Target Labels")
        axes[1].set_ylabel("Count")

        axes[2].bar(xlabels, test_targets)
        axes[2].set_title("Class-Wise Data Distribution [Test]")
        axes[2].set_xlabel("Target Labels")
        axes[2].set_ylabel("Count")

        plt.savefig(pictures_path + "data_dist.png")

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
