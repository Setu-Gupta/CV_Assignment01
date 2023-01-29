import wandb
wandb.login()

from dataset import SvnhDataset
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
import numpy as np

config = dict(
        train_ratio = 0.7,
        val_ratio = 0.2,
        test_ratio = 0.1,
        dataset = "SVNH",
        learning_rate = 0.01,
    )

# Creates the model and the data loaders
def make(config):
    # TODO
    # Create a CNN model
    model = None
    
    # Get the data from the dataset and split it into 70:20:10 chunks
    training_data, validation_data, testing_data = random_split(SvnhDataset(), [config['train_ratio'], config['val_ratio'], config['test_ratio']])

    # Create data loaders for training, validation and testing sets
    train_loader = DataLoader(training_data, shuffle=False)
    val_loader = DataLoader(validation_data, shuffle=False)
    test_loader = DataLoader(testing_data, shuffle=False)

    # Visualize the data distribution
    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    train_targets = labels = [l.item() for _, l in train_loader]
    train_ticks = np.unique(train_targets)
    train_ticks.sort()
    axes[0].hist(train_targets)
    axes[0].set_xticks(train_ticks)
    axes[0].set_title("Class-Wise Data Distribution [Train]")
    axes[0].set_xlabel("Target Labels")
    axes[0].set_ylabel("Count")
    
    val_targets = labels = [l.item() for _, l in val_loader]
    val_ticks = np.unique(val_targets)
    val_ticks.sort()
    axes[1].hist(val_targets)
    axes[1].set_xticks(val_ticks)
    axes[1].set_title("Class-Wise Data Distribution [Validation]")
    axes[1].set_xlabel("Target Labels")
    axes[1].set_ylabel("Count")
    
    test_targets = labels = [l.item() for _, l in test_loader]
    test_ticks = np.unique(test_targets)
    test_ticks.sort()
    axes[2].hist(test_targets)
    axes[2].set_xticks(test_ticks)
    axes[2].set_title("Class-Wise Data Distribution [Test]")
    axes[2].set_xlabel("Target Labels")
    axes[2].set_ylabel("Count")

    plt.savefig("./pictures/data_dist.png")

    return model, train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, config):
    # TODO
    pass

def test(model, test_loader):
    # TODO
    pass

def model_pipeline(hyperparameters):

    # Tell wandb to get started
    with wandb.init(project="CV_Assignment01", config=hyperparameters):
        # Access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Make the model, data, and optimization problem
        model, train_loader, val_loader, test_loader = make(config)

        # And use them to train the model
        train(model, train_loader, val_loader, config)

        # And test its final performance
        test(model, test_loader)

    return model

if __name__ == "__main__":
    model_pipeline(config);
