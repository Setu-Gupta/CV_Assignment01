import wandb
wandb.login()

import sys
from os.path import isfile
from dataset import SvnhDataset
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss, Softmax, Linear
from torch.optim import Adam
from torchvision import transforms
import numpy as np
import torch
import multiprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.manifold import TSNE

# Set a manual seed for reproducibility
torch.manual_seed(6225)
np.random.seed(6225)

# Set the checkpoint path
checkpoint_path = "./saved_state/custom_CNN_Q1_3.pt"

# Set the pictures directory
pictures_path = "./pictures_part3/"

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
        batch_size = 128,
        epochs = 5,
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.Normalize(mean=mean, std=std_dev) 
    ])
    
    # Load the data again but this time with transform. Split it into appropriate chunk size
    training_data, validation_data, testing_data = random_split(SvnhDataset(transform=transform), [config['train_ratio'], config['val_ratio'], config['test_ratio']])

    # Create data loaders for training, validation and testing sets
    train_loader = DataLoader(training_data, shuffle=True, batch_size=config['batch_size'], pin_memory=True)
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
    torch.onnx.export(model, input_images, "model_Q1_3.onnx")
    wandb.save("model_Q1_3.onnx")

def plot_features_tsne(model, train_loader, val_loader):
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move the model to the GPU
    model.to(device)

    # Set the node which will provide the features
    return_nodes = {"flatten" : "flatten"}

    # Create the feature extractor
    ftr_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    
    # Extract features for train and val sets
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    with torch.no_grad():
        for idx, data in enumerate(train_loader):
            # Get the input and the label
            input_images, labels = data
            
            # Save the labels
            train_labels.extend(list(labels.numpy()))

            # Move the input to the GPU
            input_images = input_images.cuda(non_blocking=True)
           
            # Extract the features
            ftrs = ftr_extractor(input_images)
            train_features.extend([x for x in ftrs['flatten'].cpu().numpy()])
        
        for idx, data in enumerate(val_loader):
            # Get the input and the label
            input_images, labels = data
            
            # Save the labels
            val_labels.extend(list(labels.numpy()))
            
            # Move the input to the GPU
            input_images = input_images.cuda(non_blocking=True)
           
            # Extract the features
            ftrs = ftr_extractor(input_images)
            val_features.extend([x for x in ftrs['flatten'].cpu().numpy()])

    # Get the set of features to perform tSNE on
    all_features = []
    all_features.extend(train_features)
    all_features.extend(val_features)
    all_labels = []
    all_labels.extend(train_labels)
    all_labels.extend(val_labels)
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    val_features = np.array(val_features)
    val_labels = np.array(val_labels)

    # Run 2D tSNE on the complete set of features
    tsne = TSNE(n_components=2, learning_rate='auto', random_state=6225, n_jobs=-1)
    all_features_2d = tsne.fit_transform(all_features)

    # Run 3D tSNE on validation features
    tsne = TSNE(n_components=3, learning_rate='auto', random_state=6225, n_jobs=-1)
    val_features_3d = tsne.fit_transform(val_features)
    
    # Set the colors and target lists
    targets = [x for x in range(10)]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
    target_names = [str(x) for x in range(10)]
    
    # Plot 2D tSNE graph for the whole set
    plt.figure()
    for target, color, label in zip(targets, colors, target_names):
        plt.scatter(all_features_2d[all_labels == target, 0], all_features_2d[all_labels == target, 1], c=colors[target], label=target_names[target])
    plt.legend()
    plt.savefig(pictures_path + 'train_val_tsne.png')

    # Plot 3D tSNE graph for the validation set
    plt.figure()
    for target, color, label in zip(targets, colors, target_names):
        plt.scatter(val_features_3d[val_labels == target, 0], val_features_3d[val_labels == target, 1], val_features_3d[val_labels == target, 2], c=colors[target], label=target_names[target])
    plt.legend()
    plt.savefig(pictures_path + 'val_tsne.png')


def model_pipeline(hyperparameters):

    # Tell wandb to get started
    with wandb.init(project="CV_Assignment01", config=hyperparameters):
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

        # Plot feature vectors using tSNE plot
        plot_features_tsne(model, train_loader, val_loader)

    return model

if __name__ == "__main__":
    model_pipeline(config)
    wandb.finish()
