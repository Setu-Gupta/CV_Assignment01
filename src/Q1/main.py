import wandb
wandb.login()

from dataset import SvnhDataset
from torch.utils.data import DataLoader, random_split

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
    train_loader = DataLoader(training_data, shuffle=True)
    val_loader = DataLoader(validation_data, shuffle=True)
    test_loader = DataLoader(testing_data, shuffle=True)

    # TODO
    # Visualize the data distribution

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
