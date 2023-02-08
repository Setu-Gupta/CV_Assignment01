import wandb
wandb.login()

import sys
from os.path import isfile
from dataset import SvnhDataset
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split, ConcatDataset
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss, Softmax, Linear
from torch.optim import Adam
from torchvision import transforms
import numpy as np
import torch
import multiprocessing
from copy import deepcopy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearnex import patch_sklearn

# Use intel MKL for sklearn
patch_sklearn()

# Set a manual seed for reproducibility
torch.manual_seed(6225)
np.random.seed(6225)

# Set the checkpoint path
checkpoint_path = "./saved_state/Resnet_Q1_4.pt"

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
        weight_decay = 0.1,
        batch_size = 200,
        epochs = 2,
        log_interval = 10
    )

# Get the predicted label
# Predict the most probable label
softmax = Softmax(dim=1).to(device)
def predict_label(x):
    probs = softmax(x)
    return torch.argmax(probs, dim=1)

# Creates the model and the data loaders
def make(config):
    # Get the Resnet18 model with pre-trained weights
    model = resnet18(weights='DEFAULT')

    # Resize the fully-connected classification layer
    fc_in_size = model.fc.in_features
    model.fc = Linear(in_features=fc_in_size, out_features=10)

    # Set the input image size on which Resnet was trained
    input_size = 224

    # Create the loss criterion
    loss_criterion = CrossEntropyLoss()
   
    # Create the optimizer
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], betas=(config['adam_beta1'], config['adam_beta2']), weight_decay=config['weight_decay'])

    # Create the transform to normalize the images
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std_dev = torch.Tensor([0.229, 0.224, 0.225])
    transform_none = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.Normalize(mean=mean, std=std_dev) 
    ])

    # Create the transform to normalize the images and add color jitter
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std_dev = torch.Tensor([0.229, 0.224, 0.225])
    transform_color = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(),
        transforms.Resize(input_size),
        transforms.Normalize(mean=mean, std=std_dev)
    ])
    
    # Create the transform to normalize the images and apply a random affine transformation
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std_dev = torch.Tensor([0.229, 0.224, 0.225])
    transform_affine = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.7, 1.3)),
        transforms.Resize(input_size),
        transforms.Normalize(mean=mean, std=std_dev)
    ])
    
    # Create the transform to normalize the images and apply gaussian blur
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std_dev = torch.Tensor([0.229, 0.224, 0.225])
    transform_blur = transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(3),
        transforms.Resize(input_size),
        transforms.Normalize(mean=mean, std=std_dev)
    ])

    # Split the dataset into appropriate chunk size
    training_data, validation_data, testing_data = random_split(SvnhDataset(transform=transform_none), [config['train_ratio'], config['val_ratio'], config['test_ratio']])

    # Make 3 copies of training data for augmentation
    training_data_color = deepcopy(training_data)
    training_data_affine = deepcopy(training_data)
    training_data_blur = deepcopy(training_data)

    # Set the transforms for the three copies
    training_data_color.transform=transform_color
    training_data_affine.transform=transform_affine
    training_data_blur.transform=transform_blur

    # Concatenate the training dataset with all the four transforms
    training_data_full = ConcatDataset([training_data, training_data_color, training_data_affine, training_data_blur])

    # Create data loaders for training, validation and testing sets
    train_loader = DataLoader(training_data_full, shuffle=True, batch_size=config['batch_size'], pin_memory=True)
    val_loader = DataLoader(validation_data, shuffle=True, batch_size=config['batch_size'], pin_memory=True)
    test_loader = DataLoader(testing_data, shuffle=True, batch_size=config['batch_size'], pin_memory=True)

    return model, loss_criterion, optimizer, train_loader, val_loader, test_loader

def train(model, loss_criterion, optimizer, train_loader, val_loader, config):
   
    # Epoch to start training from
    start_epoch = 0

    # If a model is saved and check-pointing is enabled, load its state
    if("save" in sys.argv):
        if(isfile(checkpoint_path)):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1

    # Set the model to training mode
    model.train()

    # Move the model to GPU
    model.to(device)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_criterion, log="all", log_freq=config['log_interval'])

    for epoch_count in range(start_epoch, config['epochs']):
        
        # Track the cumulative loss
        running_loss = 0.0
        running_count = 0
        train_ground_truth = []
        train_preds = []
        for idx, data in enumerate(train_loader):

            # Get the inputs and labels
            input_images, labels = data
                        
            # Append true labels to ground truth
            train_ground_truth.extend(list(labels.numpy()))

            # Move the input and output to the GPU
            input_images = input_images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Zero out the gradient of the optimizer
            optimizer.zero_grad()
            
            # Feed the input to the network
            predictions = model(input_images)
            
            # Compute the loss
            loss = loss_criterion(predictions, labels)

            # Append predicted labels to preds
            with torch.no_grad():
                outputs = model(input_images)
                pred_labels = predict_label(outputs)
                train_preds.extend(list(pred_labels.cpu().numpy()))

            # Perform back propagation
            loss.backward()

            # Take a step towards the minima
            optimizer.step()
           
            # Update training loss
            running_loss += loss.item()
            running_count += config['batch_size']

            # Log validation and training loss with wandb
            if idx % config['log_interval'] == 0 and idx != 0:
                # Set the model to evaluation mode
                model.eval()
                
                # Compute validation loss
                val_loss = 0.0
                val_preds = []
                val_ground_truth = []
                with torch.no_grad():
                    for data in val_loader:
                        # Get the input and the label
                        input_images, labels = data
                        
                        # Append true labels to ground truth
                        val_ground_truth.extend(list(labels.numpy()))
                       
                        # Move the input and output to the GPU
                        input_images = input_images.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)
                        
                        # Get prediction and compute loss
                        predictions = model(input_images)
                        val_loss += loss_criterion(predictions, labels).item()
            
                        # Append predicted labels to preds
                        outputs = model(input_images)
                        pred_labels = predict_label(outputs)
                        val_preds.extend(list(pred_labels.cpu().numpy()))
                
                val_loss /= len(val_loader.dataset)
                train_loss = running_loss/running_count 
                val_accu = accuracy_score(val_ground_truth, val_preds, normalize=True)
                train_accu = accuracy_score(train_ground_truth, train_preds, normalize=True)
                print(f"[epoch:{epoch_count+1}, iteration:{idx}] Average Training Loss: {train_loss}, Average Validation Loss: {val_loss}")
                running_loss = 0.0
                running_count = 0
                train_ground_truth = []
                train_preds = []
                
                wandb.log({"epoch": epoch_count + 1, "train_loss": train_loss, "validation_loss": val_loss, "training accuracy": train_accu, "validation accuracy": val_accu})
    
                # Reset the model back to training mode
                model.train()

        # If check-pointing is enabled, save current state
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

    # Set the model to evaluation mode
    model.eval()

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
            outputs = model(input_images)
            pred_labels = predict_label(outputs)
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
    torch.onnx.export(model, input_images, "model_Q1_4.onnx")
    wandb.save("model_Q1_4.onnx")

def model_pipeline(hyperparameters):

    # Tell wandb to get started
    with wandb.init(project="CV_Assignment01", config=hyperparameters, name="Data Augmented Resnet"):
        print("The model will be running on", device, "device")

        # Access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Make the model, data loader, the loss criterion and the optimizer
        model, loss_criterion, optimizer, train_loader, val_loader, test_loader = make(config)

        # And use them to train the model
        if "test" not in sys.argv:
            train(model, loss_criterion, optimizer, train_loader, val_loader, config)

        # And test its final performance
        test(model, test_loader)

    return model

if __name__ == "__main__":
    model_pipeline(config)
    wandb.finish()
