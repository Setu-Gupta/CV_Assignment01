import wandb
wandb.login()

from os.path import isfile
from dataset import SvnhDataset
from network import Net 
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch
from torch.nn import Linear, Softmax
import multiprocessing
import csv

# Set a manual seed for reproducibility
torch.manual_seed(6225)
np.random.seed(6225)

# Set the checkpoint path
checkpoint_path_1 = "./saved_state/custom_CNN_Q1_2.pt"
checkpoint_path_2 = "./saved_state/Resnet_Q1_3.pt"
checkpoint_path_3 = "./saved_state/Resnet_Q1_4.pt"

# Set the output CSV path
csv_path = "./Q1_5.csv"

# Use all cores
torch.set_num_threads(multiprocessing.cpu_count())

# Use GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The configuration for wandb
config = dict(
        dataset = "SVNH",
        batch_size = 700
    )

# Creates the models and the data loaders
def make(config):

    # Create and load model1 i.e. Custom CNN
    model1 = Net()
    if(isfile(checkpoint_path_1)):
        checkpoint = torch.load(checkpoint_path_1)
        model1.load_state_dict(checkpoint['model'])
    else:
        print("Could not load Model-1, checkpoint file not found")

    # Create and load model2 i.e. Resnet18
    model2 = resnet18(weights='DEFAULT')
    fc_in_size = model2.fc.in_features
    model2.fc = Linear(in_features=fc_in_size, out_features=10)
    if(isfile(checkpoint_path_2)):
        checkpoint = torch.load(checkpoint_path_2)
        model2.load_state_dict(checkpoint['model'])
    else:
        print("Could not load Model-2, checkpoint file not found")
    model2.eval()

    # Create and load model3 i.e. Resnet18 with augmented data
    model3 = resnet18(weights='DEFAULT')
    fc_in_size = model3.fc.in_features
    model3.fc = Linear(in_features=fc_in_size, out_features=10)
    if(isfile(checkpoint_path_3)):
        checkpoint = torch.load(checkpoint_path_3)
        model3.load_state_dict(checkpoint['model'])
    else:
        print("Could not load Model-3, checkpoint file not found")
    model3.eval()

    # Create the feature extractors for resnet18
    return_nodes = {'flatten' : 'flatten' }
    extractor2 = create_feature_extractor(model2, return_nodes)
    extractor3 = create_feature_extractor(model2, return_nodes)

    # Get the mean and standard deviation of the dataset
    with torch.no_grad():   # Ensure that no intermediate computations get stored onto the GPU
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
 
    # Create the transform for Custom CNN
    transform_custom_CNN = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev)
    ])
    
    # Create the transform for Custom CNN to Resnet18 conversion
    input_size = 224
    resnet_mean = torch.Tensor([0.485, 0.456, 0.406])
    resnet_std_dev = torch.Tensor([0.229, 0.224, 0.225])
    transform_resnet18 = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=1/std_dev),
        transforms.Normalize(mean=-mean, std=[1., 1., 1.]),
        transforms.Resize(input_size),
        transforms.Normalize(mean=resnet_mean, std=resnet_std_dev) 
    ])

    # Create the data loader
    data_loader = DataLoader(SvnhDataset(transform=transform_custom_CNN), shuffle=True, batch_size=config['batch_size'], pin_memory=True)

    # Create a list of models
    models = [model1, model2, model3]

    # Create a list of extractors
    extractors = [extractor2, extractor3]

    return models, extractors, data_loader, transform_resnet18

def get_means(models, extractors, data_loader, transform_resnet18):
    # Create the tensor to store the means and counts of vectors in each class for each model
    model1_means = torch.zeros((10, 65536)).to(device)
    model2_means = torch.zeros((10, 512)).to(device)
    model3_means = torch.zeros((10, 512)).to(device)
    counts = torch.zeros((10)).to(device)

    with torch.no_grad():   # Ensure that no intermediate computations get stored onto the GPU
        # For each model, compute the sums and counts
        for model_idx, model in enumerate([models[0], extractors[0], extractors[1]]):
            
            # Move the model onto the GPU
            model.to(device)

            # Compute sums and counts
            for images, labels in data_loader:
                
                # Move the images and the labels to the GPU
                images = images.to(device)
                labels = labels.to(device)

                # If model1 is being used, compute the features directly
                if(model_idx == 0):
                    outputs = model.get_features(images)
                
                # If model2 or model3 is being used, apply transform for resnet18 and compute the features
                if(model_idx > 0):
                    resnet_images = torch.zeros((images.shape[0], 3, 224, 224)).to(device)
                    for idx, image in enumerate(images):
                        resnet_images[idx] = transform_resnet18(image)
                    outputs = model(resnet_images)['flatten']

                # Update counts and means for this batch
                for i in range(labels.shape[0]):
                    o = outputs[i]
                    l = labels[i]
                    counts[l.item()] += 1
                    if(model_idx == 0):
                        model1_means[l.item()] += o
                    elif(model_idx == 1):
                        model2_means[l.item()] += o
                    elif(model_idx == 2):
                        model3_means[l.item()] += o

        # Compute means by dividing the cumulative sum stored in the means tensor by the count
        for i in range(10):
            model1_means[i] = model1_means[i] / counts[i]    
            model2_means[i] = model2_means[i] / counts[i]    
            model3_means[i] = model3_means[i] / counts[i]    
   
    # Store all means in a list
    means = [model1_means, model2_means, model3_means]

    return means

# Predict the most probable label
softmax = Softmax(dim=1).to(device)
def predict_label(x):
    probs = softmax(x)
    return torch.argmax(probs, dim=1)

def get_mispreds(models, extractors, data_loader, transform_resnet18):
    
    # Store the misprediction counts
    model1_mispred_features = torch.zeros((10, 3, 65536))   # 10 classes with 3 mispredictions each represented via a vector of length of 65536
    model2_mispred_features = torch.zeros((10, 3, 512))     # 10 classes with 3 mispredictions each represented via a vector of length of 512
    model3_mispred_features = torch.zeros((10, 3, 512))     # 10 classes with 3 mispredictions each represented via a vector of length of 512
    mispred_counts = torch.zeros((10), dtype=int)           # Number of mispredictions for each class
    mispred_predictions_and_truth = torch.zeros((10, 3), dtype=int) # 10 classes, 3 mispredictions with 1 predicted label

    # Move everything to the GPU
    model1_mispred_features = model1_mispred_features.to(device)
    model2_mispred_features = model2_mispred_features.to(device)
    model3_mispred_features = model3_mispred_features.to(device)
    mispred_counts = mispred_counts.to(device)
    mispred_predictions_and_truth = mispred_predictions_and_truth.to(device)
   
    # Move the models onto the GPU
    for i in range(3):
        models[i] = models[i].to(device)

    # Get 3 mispredictions for each class for model1
    with torch.no_grad():   # Ensure that no intermediate computations get stored onto the GPU
        for images, labels in data_loader:
            # Move the images and the labels to the GPU
            images = images.to(device)
            labels = labels.to(device)

            # Get the predictions
            preds = models[0].predict_label(images)
          
            # Iterate over images in the batch to check for mispredictions
            for idx in range(labels.shape[0]):
                p = preds[idx]
                l = labels[idx]
                i = images[idx]
                # Check for misprediction
                current_count = mispred_counts[l.item()].item()
                if(p.item() != l.item() and current_count < 3):
                    mispred_counts[l.item()] += 1
                    
                    # Store model1 features
                    model1_features = models[0].get_features(i).flatten()
                    model1_mispred_features[l.item()][current_count] = model1_features
                    mispred_predictions_and_truth[l.item()][current_count] = p.item()
                    
                    # Apply transform on the image to pass it to resnet18
                    resnet_image = transform_resnet18(i)
                    resnet_image = resnet_image[None]   # Add extra dummy dimension
                    
                    # Store model2 features
                    model2_out = models[1](resnet_image)
                    model2_pred = predict_label(model2_out)
                    if model2_pred.item() != l.item():
                        model2_features = extractors[0](resnet_image)['flatten']
                        model2_mispred_features[l.item()][current_count] = model2_features

                    # Store model3 features
                    model3_out = models[2](resnet_image)
                    model3_pred = predict_label(model3_out)
                    if model3_pred.item() != l.item():
                        model3_features = extractors[1](resnet_image)['flatten']
                        model3_mispred_features[l.item()][current_count] = model3_features

            # Exit if 3 mispredictions found for all classes
            should_exit = True
            for count in mispred_counts:
                if count.item() < 3:
                    should_exit = False
            
            if(should_exit):
                break
    
    # Store all mispredcition features in a list
    mispreds = [model1_mispred_features, model2_mispred_features, model3_mispred_features]
    
    return mispreds, mispred_predictions_and_truth

def get_distances(mispreds, means, mispred_predictions_and_truth):
    
    # Create distances tensor to store the euclidean distances
    distances = torch.zeros((3, 10, 3, 2))  # 3 models, 10 classes, 3 mispredictions, 2 distances (truth and predicted)

    # Move distances to GPU
    distances = distances.to(device)

    # Compute the distances
    with torch.no_grad():   # Ensure that no intermediate computations get stored onto the GPU
        for model_idx in range(3):
            for class_idx in range(10):
                for mispred_idx in range(3):

                    # Get the true and predicted label
                    true_label = class_idx
                    pred_label = mispred_predictions_and_truth[class_idx][mispred_idx].item()
                    
                    # Get the means of the predicted and true class
                    true_mean = means[model_idx][true_label]
                    pred_mean = means[model_idx][pred_label]

                    # Get the predicted vector
                    pred_vector = mispreds[model_idx][class_idx][mispred_idx]
                    
                    # Compute the distances
                    distances[model_idx][class_idx][mispred_idx][0] = (pred_vector - true_mean).pow(2).sum().sqrt()
                    distances[model_idx][class_idx][mispred_idx][1] = (pred_vector - pred_mean).pow(2).sum().sqrt()
    
    return distances

def dump_distances(distances, mispred_predictions_and_truth):
    
    # Write the CSV header
    row = ["True Label",
           "Predicted Label",
           "Euclidean distance from true class for CustomCNN",
           "Euclidean distance from predicted class for CustomCNN",
           "Euclidean distance from true class for Resnet18",
           "Euclidean distance from predicted class for Resnet18",
           "Euclidean distance from true class for data augmented Resnet18",
           "Euclidean distance from predicted class for data augmented Resnet18",
        ]
    with open(csv_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(row)

    # Dump all the data 
    for class_idx in range(10):
        for mispred_idx in range(3):
            true_label = class_idx
            pred_label = mispred_predictions_and_truth[class_idx][mispred_idx].item()
            row = [true_label, pred_label]

            for model_idx in range(3):
                true_distance = distances[model_idx][class_idx][mispred_idx][0].item()
                pred_distance = distances[model_idx][class_idx][mispred_idx][1].item()

                if pred_label == true_label:
                    row.append("NA")
                    row.append("NA")
                else:
                    row.append(true_distance)
                    row.append(pred_distance)

            with open(csv_path, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(row)

def analyze_distance(hyperparameters):

    # Tell wandb to get started
    with wandb.init(project="CV_Assignment01", config=hyperparameters, name="Distance Analysis"):
        print("The model will be running on", device, "device")

        # Access all hyperparameters through wandb.config, so logging matches execution!
        config = wandb.config

        # Make the models and the data loaders
        print("Creating models and data loader")
        models, extractors, data_loader, transform_resnet18 = make(config)

        # Compute mean vectors for all three models
        print("Computing means")
        means = get_means(models, extractors, data_loader, transform_resnet18)
        
        # Compute the misprediction vectors indices for all three models
        print("Computing mispredictions")
        mispreds, mispred_predictions_and_truth = get_mispreds(models, extractors, data_loader, transform_resnet18) 

        # Compute euclidean distances
        print("Computing distances")
        distances = get_distances(mispreds, means, mispred_predictions_and_truth)
        distances = distances.cpu()
        mispred_predictions_and_truth = mispred_predictions_and_truth.cpu()

        # Save the distances in a CSV file
        print("Dumping data to " + csv_path)
        dump_distances(distances, mispred_predictions_and_truth)

if __name__ == "__main__":
    analyze_distance(config)
    wandb.finish()
