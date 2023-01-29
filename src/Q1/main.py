import wandb
wandb.login()

from dataset import SvnhDataset
from network import Net 
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam 
import numpy as np
import torch
import multiprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Use all cores
torch.set_num_threads(multiprocessing.cpu_count())

# Use GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")

config = dict(
        train_ratio = 0.7,
        val_ratio = 0.2,
        test_ratio = 0.1,
        dataset = "SVNH",
        adam_beta1 = 0.9, 
        adam_beta2 = 0.999,
        learning_rate = 0.001,
        epochs = 10,
        log_interval = 100
    )

# Creates the model and the data loaders
def make(config):
    # Create a CNN model
    model = Net()

    # Create the loss criterion
    loss_criterion = CrossEntropyLoss()
    
    # Create the optimizer
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], betas=(config['adam_beta1'], config['adam_beta2']))

    # Get the data from the dataset and split it into 70:20:10 chunks
    training_data, validation_data, testing_data = random_split(SvnhDataset(), [config['train_ratio'], config['val_ratio'], config['test_ratio']])

    # Create data loaders for training, validation and testing sets
    train_loader = DataLoader(training_data, shuffle=False, pin_memory=True)
    val_loader = DataLoader(validation_data, shuffle=False, pin_memory=True)
    test_loader = DataLoader(testing_data, shuffle=False, pin_memory=True)

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

    return model, loss_criterion, optimizer, train_loader, val_loader, test_loader

def train(model, loss_criterion, optimizer, train_loader, val_loader, config):
    
    # Move the model to GPU
    model.to(device)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_criterion, log="all", log_freq=config['log_interval'])

    for epoch_count in range(config['epochs']):
        
        # Track the cumulative loss
        running_loss = 0.0
        for idx, data in enumerate(train_loader):

            # Get the inputs and labels
            input_image, _label = data

            # Convert _label into a probabilty vector
            label = np.zeros(10)
            label[_label.item() - 1] = 1.0
            label = torch.from_numpy(label)
            
            # Move the input and output to the GPU
            input_image = input_image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # Zero out the gradient of the optimizer
            optimizer.zero_grad()
            
            # Feed the input to the network
            prediction = model(input_image)

            # Compute the loss
            loss = loss_criterion(prediction, label)

            # Perform back propogation
            loss.backward()

            # Take a step towards the minima
            optimizer.step()
            
            running_loss += loss.item()
           
            # Log validation and training loss with wandb
            if idx % config['log_interval'] == 0 and idx != 0:
                val_loss = 0.0
                with torch.no_grad():
                    for data in val_loader:
                        # Get the input and the label
                        input_image, _label = data
                       
                        # Get the true label in pytorch format
                        label = np.zeros(10)
                        label[_label.item() - 1] = 1.0
                        label = torch.from_numpy(label)
                        
                        # Move the input and output to the GPU
                        input_image = input_image.cuda(non_blocking=True)
                        label = label.cuda(non_blocking=True)
                        
                        # Get prediction and compute loss
                        prediction = model(input_image)
                        val_loss += loss_criterion(prediction, label)
                
                val_loss /= len(val_loader)
                train_loss = running_loss/config['log_interval']
                print(f"[epoch:{epoch_count+1}, iteration:{idx}] Average Training Loss: {train_loss}, Average Validation Loss: {val_loss}")
                running_loss = 0.0
                
                wandb.log({"epoch": epoch_count + 1, "train_loss": train_loss, "validation_loss": val_loss})

def test(model, test_loader):
    
    # Move the model to the GPU
    model.to(device)

    # Create an array of ground truth labels and prediction probabilities
    ground_truth = np.zeros(len(test_loader))
    
    # Create an array of predicted labels and their probabilities
    preds = np.ones(len(test_loader))
    pred_probs = []

    # Create an array of class names
    class_names = [str(x) for x in range(10)]

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # Get the input and the label
            input_image, label = data
            
            # Move the input to the GPU
            input_image = input_image.cuda(non_blocking=True)
            
            # Store the ground truth
            ground_truth[idx] = label

            # Get prediction probabilities
            prediction_proba = model(input_image)
            pred_probs.append(prediction_proba.cpu().numpy())

            # Get predicted label
            pred_label = np.argmax(prediction_proba.cpu().numpy()) + 1
            preds[idx] = pred_label

    # Convert to numpy arrays
    pred_probs = np.asarray(pred_probs)

    # Compute accuracy, precision, recall and f1_score
    accuracy = accuracy_score(ground_truth, preds)
    precision = precision_score(ground_truth, preds)
    recall = recall_score(ground_truth, preds)
    accuracy = accuracy_score(ground_truth, preds)

    # Log the confusion matrix
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=pred_probs, y_true=ground_truth, class_names=class_names),
               "F1_score": f1_score,
               "Accuracy": accuracy,
               "Precision": precision,
               "Recall": recall})

    # Save the model
    torch.onnx.export(model, input_image, "model.onnx")
    wandb.save("model.onnx")

def analyze_misclassifications(model, test_loader):
    # Move the model to the GPU
    model.to(device)

    visualization_count = [0 for i in range(10)]
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # Get the input and the label
            input_image, label = data
            
            # Move the input to GPU
            input_image = input_image.cuda(non_blocking=True)

            # Get prediction probabilities
            prediction_proba = model(input_image)

            # Get predicted label
            pred_label = np.argmax(prediction_proba.cpu().numpy()) + 1
            
            # Check if a misprediction ouccurred
            if(pred_label != label and visualization_count[label - 1] < 3):
                # Increment the visualization count
                visualization_count[label - 1] += 1
                
                # Save the mispredicted image
                input_image = np.transpose(input_image, axes=[1, 2, 0])
                plt.imshow(input_image)
                img_name = str(label) + '_' + str(visualization_count[label - 1]) + '_' + str(pred_label)
                plt.savefig('./pictures/' + img_name)

            # Exit if at least 3 images for all classes have been saved
            should_exit = True
            for val in visualization_count:
                if(val < 3):
                    should_exit = False

            if(should_exit):
                break

def model_pipeline(hyperparameters):

    # Tell wandb to get started
    with wandb.init(project="CV_Assignment01", config=hyperparameters):
        # Access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Make the model, data loader, the loss criterion and the optimizer
        model, loss_criterion, optimizer, train_loader, val_loader, test_loader = make(config)

        # And use them to train the model
        train(model, loss_criterion, optimizer, train_loader, val_loader, config)

        # And test its final performance
        test(model, test_loader)

        # Perform miss-classification analysis
        analyze_misclassifications(model, test_loader)

    return model

if __name__ == "__main__":
    model_pipeline(config);
