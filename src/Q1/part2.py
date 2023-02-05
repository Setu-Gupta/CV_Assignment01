import wandb
wandb.login()

import sys
from os.path import isfile
from dataset import SvnhDataset
from network import Net 
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
import numpy as np
import torch
import multiprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Set a manual seed for reproducibility
torch.manual_seed(6225)
np.random.seed(6225)

# Set the checkpoint path
checkpoint_path = "./saved_state/custom_CNN_Q1_2.pt"

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
        learning_rate = 0.001,
        weight_decay = 0.005,
        batch_size = 32,
        epochs = 3,
        log_interval = 10
    )

# Creates the model and the data loaders
def make(config):
    # Create a CNN model
    model = Net()

    # Create the loss criterion
    loss_criterion = CrossEntropyLoss()
    
    # Create the optimizer
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], betas=(config['adam_beta1'], config['adam_beta2']), weight_decay=config['weight_decay'])

    # Get the mean and standard deviation of the dataset
    all_data = DataLoader(SvnhDataset(transform=transforms.ToTensor()), shuffle=True, batch_size=config['batch_size'], pin_memory=True)
    channel_sum = torch.zeros(3).to(device)
    channel_sum_squared = torch.zeros(3).to(device)
    num_batches = torch.Tensor([0])
    num_batches = num_batches.to(device)
    for images, label in all_data:
        channel_sum += torch.mean(images)
        channel_sum_squared += torch.mean(torch.square(images))
        num_batches += 1

    mean = channel_sum/num_batches
    std_dev = torch.sqrt(((channel_sum_squared/num_batches) - torch.square(mean)))
    mean = mean
    std_dev = std_dev
 
    # Create the transform to normalize the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev)
        ])
    
    # Create the transform to invese normalize the images
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=1/std_dev),
        transforms.Normalize(mean=-mean, std=[1., 1., 1.]),
        transforms.ToPILImage()
        ])

    # Load the data again but this time with transform. Split it into appropriate chunk size
    training_data, validation_data, testing_data = random_split(SvnhDataset(transform=transform), [config['train_ratio'], config['val_ratio'], config['test_ratio']])

    # Create data loaders for training, validation and testing sets
    train_loader = DataLoader(training_data, shuffle=True, batch_size=config['batch_size'], pin_memory=True)
    val_loader = DataLoader(validation_data, shuffle=True, batch_size=config['batch_size'], pin_memory=True)
    test_loader = DataLoader(testing_data, shuffle=True, batch_size=config['batch_size'], pin_memory=True)

    # Visualize the data distribution
    # Get all the targets
    train_targets = []
    val_targets = []
    test_targets = []
    for _, labels in train_loader:
        labels = labels.numpy()
        labels = labels.flatten()
        train_targets.extend(list(labels))
    for _, labels in val_loader:
        labels = labels.numpy()
        labels = labels.flatten()
        val_targets.extend(list(labels))
    for _, labels in test_loader:
        labels = labels.numpy()
        labels = labels.flatten()
        test_targets.extend(list(labels))

    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    
    train_ticks = np.unique(train_targets)
    train_ticks.sort()
    axes[0].hist(train_targets)
    axes[0].set_xticks(train_ticks)
    axes[0].set_title("Class-Wise Data Distribution [Train]")
    axes[0].set_xlabel("Target Labels")
    axes[0].set_ylabel("Count")
    
    val_ticks = np.unique(val_targets)
    val_ticks.sort()
    axes[1].hist(val_targets)
    axes[1].set_xticks(val_ticks)
    axes[1].set_title("Class-Wise Data Distribution [Validation]")
    axes[1].set_xlabel("Target Labels")
    axes[1].set_ylabel("Count")
    
    test_ticks = np.unique(test_targets)
    test_ticks.sort()
    axes[2].hist(test_targets)
    axes[2].set_xticks(test_ticks)
    axes[2].set_title("Class-Wise Data Distribution [Test]")
    axes[2].set_xlabel("Target Labels")
    axes[2].set_ylabel("Count")

    plt.savefig(pictures_path + "data_dist.png")

    return model, loss_criterion, optimizer, train_loader, val_loader, test_loader, inv_transform

def train(model, loss_criterion, optimizer, train_loader, val_loader, config):
   
    # Epoch to start training from
    start_epoch = 0

    # If a model is saved and checkpointing is enabled, load its state
    if("save" in sys.argv):
        if(isfile(checkpoint_path)):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1

    # Move the model to GPU
    model.to(device)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_criterion, log="all", log_freq=config['log_interval'])

    for epoch_count in range(start_epoch, config['epochs']):
        
        # Track the cumulative loss
        running_loss = 0.0
        running_count = 0
        for idx, data in enumerate(train_loader):

            # Get the inputs and labels
            input_images, labels = data

            # Move the input and output to the GPU
            input_images = input_images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Zero out the gradient of the optimizer
            optimizer.zero_grad()
            
            # Feed the input to the network
            predictions = model(input_images)
            
            # Compute the loss
            loss = loss_criterion(predictions, labels)

            # Perform back propogation
            loss.backward()

            # Take a step towards the minima
            optimizer.step()
           
            # Update training loss
            running_loss += loss.item()
            running_count += config['batch_size']

            # Log validation and training loss with wandb
            if idx % config['log_interval'] == 0 and idx != 0:
                val_loss = 0.0
                with torch.no_grad():
                    for data in val_loader:
                        # Get the input and the label
                        input_images, labels = data
                       
                        # Move the input and output to the GPU
                        input_images = input_images.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)
                        
                        # Get prediction and compute loss
                        predictions = model(input_images)
                        val_loss += loss_criterion(predictions, labels).item()
                
                val_loss /= len(val_loader.dataset)
                train_loss = running_loss/running_count 
                print(f"[epoch:{epoch_count+1}, iteration:{idx}] Average Training Loss: {train_loss}, Average Validation Loss: {val_loss}")
                running_loss = 0.0
                running_count = 0
                
                wandb.log({"epoch": epoch_count + 1, "train_loss": train_loss, "validation_loss": val_loss})
    
        # If checkpointing is enabled, save current state
        if("save" in sys.argv):
            torch.save({
                'model'     : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch'     : epoch_count
                }, checkpoint_path)
    
    # Save the model on wandb
    if("save" in sys.argv):
        model_artifact = wandb.Artifact('model', type='model')
        model_artifact.add_file(checkpoint_path)
        wandb.log_artifact(model_artifact)

def test(model, test_loader):
   
    # Load the saved model
    if("save" in sys.argv):
        if(isfile(checkpoint_path)):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])

    # Move the model to the GPU
    model.to(device)

    # Create an array of ground truth labels
    ground_truth = []
    
    # Create an array of predicted labels
    preds = []

    # Create an array of class names
    class_names = [str(x) for x in range(10)]

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # Get the input and the label
            input_images, labels = data
            
            # Move the input to the GPU
            input_images = input_images.cuda(non_blocking=True)
            
            # Store the ground truth
            ground_truth.extend(list(labels.numpy()))

            # Get predicted label
            pred_labels = model.predict_label(input_images)
            preds.extend(list(pred_labels.cpu().numpy()))

    # Compute accuracy, precision, recall and f1_score
    accuracy = accuracy_score(ground_truth, preds, normalize=True)
    precision = precision_score(ground_truth, preds, average='weighted', zero_division=1)
    recall = recall_score(ground_truth, preds, average='weighted', zero_division=1)
    f_score = f1_score(ground_truth, preds, average='weighted', zero_division=1)

    # Log the confusion matrix
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(preds=preds, y_true=ground_truth, class_names=class_names),
               "F1_score": f_score,
               "Accuracy": accuracy,
               "Precision": precision,
               "Recall": recall})

    # Save the model
    torch.onnx.export(model, input_images, "model_Q1_2.onnx")
    wandb.save("model_Q1_2.onnx")

def analyze_misclassifications(model, test_loader, inv_transform):
    # Move the model to the GPU
    model.to(device)

    visualization_count = [0 for i in range(10)]
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # Get the input and the label
            input_images, labels = data
            
            # Move the input to GPU
            input_images = input_images.cuda(non_blocking=True)

            # Get predicted label
            pred_labels = model.predict_label(input_images) 
            
            # Check if a misprediction ouccurred and three images have not been saved yet
            for input_image, pred_label, label in zip(input_images.cpu(), pred_labels.cpu(), labels.cpu()):
                pred_label = pred_label.item()
                label = label.item()

                if(pred_label != label and visualization_count[label] < 3):
                    # Increment the visualization count
                    visualization_count[label] += 1

                    # Apply the inverse of the transform
                    input_image = inv_transform(input_image)

                    # Save the mispredicted image
                    plt.figure()
                    plt.imshow(input_image)
                    img_name = 'true_' + str(label) + '_' + str(visualization_count[label]) + '_pred_' + str(pred_label)
                    plt.savefig(pictures_path + img_name)

                # Exit if at least 3 images for all classes have been saved
                should_exit = True
                for val in visualization_count:
                    if(val < 3):
                        should_exit = False

                if(should_exit):
                    return

def model_pipeline(hyperparameters):

    # Tell wandb to get started
    with wandb.init(project="CV_Assignment01", config=hyperparameters):
        print("The model will be running on", device, "device")

        # Access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Make the model, data loader, the loss criterion and the optimizer
        model, loss_criterion, optimizer, train_loader, val_loader, test_loader, inv_transform = make(config)

        # And use them to train the model
        if "test" not in sys.argv:
            train(model, loss_criterion, optimizer, train_loader, val_loader, config)

        # And test its final performance
        test(model, test_loader)

        # Perform miss-classification analysis
        analyze_misclassifications(model, test_loader, inv_transform)

    return model

if __name__ == "__main__":
    model_pipeline(config)
    wandb.finish()
