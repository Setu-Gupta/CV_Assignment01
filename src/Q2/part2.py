import wandb
wandb.login()

import sys
from os.path import isfile
from dataset import VocDataset
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torch.optim import Adam
import torch
import multiprocessing
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, accuracy_score

# Set a manual seed for reproducibility
torch.manual_seed(6225)
np.random.seed(6225)

# Set the checkpoint path
checkpoint_path = "./saved_state/fcn_resnet50_Q2_2.pt"

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
        dataset = "VOC",
        dim = 300,
        adam_beta1 = 0.9, 
        adam_beta2 = 0.999,
        learning_rate = 0.01,
        weight_decay = 0.005,
        batch_size = 15,
        epochs = 1000,
        log_interval = 1,
    )

# Custom transform for masks
class MyToTensor(object):
    """Convert masks to Tensors."""

    def __call__(self, mask):
        return torch.from_numpy(mask.astype('int64'))

# Creates the model and the data loaders
def make(config):
    # Create the model
    model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)

    # Create the loss criterion
    loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    
    # Create the optimizer
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], betas=(config['adam_beta1'], config['adam_beta2']), weight_decay=config['weight_decay'])

    # Create the transform to resize the images
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['dim'], config['dim']), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Normalize(mean=mean, std=std_dev),
    ])
    
    # Create the transform to resize the masks
    transform_mask = transforms.Compose([
        MyToTensor(),
        transforms.Resize((config['dim'], config['dim']), interpolation=transforms.InterpolationMode.NEAREST),
    ])
    
    # Create the transform to inverse normalize the images
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=1/std_dev),
        transforms.Normalize(mean=-mean, std=[1., 1., 1.]),
        transforms.ToPILImage(),
    ])

    # Load the data again but this time with transform. Split it into appropriate chunk size
    training_data, validation_data, testing_data = random_split(VocDataset(transform_image=transform_image, transform_mask=transform_mask), [config['train_ratio'], config['val_ratio'], config['test_ratio']])

    # Create data loaders for training, validation and testing sets
    train_loader = DataLoader(training_data, shuffle=True, pin_memory=True, batch_size=config['batch_size'], num_workers=multiprocessing.cpu_count())
    val_loader = DataLoader(validation_data, shuffle=True, pin_memory=True, batch_size=config['batch_size'], num_workers=multiprocessing.cpu_count())
    test_loader = DataLoader(testing_data, shuffle=True, pin_memory=True, batch_size=config['batch_size'], num_workers=multiprocessing.cpu_count())

    # Visualize the data distribution
    # Get the target labels
    with torch.no_grad():   # Ensure that no intermidiate results are stored on the GPU
        train_targets = torch.zeros((21), dtype=float).to(device)
        val_targets = torch.zeros((21), dtype=float).to(device) 
        test_targets = torch.zeros((21), dtype=float).to(device)
       
        for _, masks in train_loader:
            masks = masks.to(device)
            labels = torch.flatten(masks).to(device)
            for idx in range(21):
                train_targets[idx] = torch.numel(labels[labels==idx])

        for _, masks in val_loader:
            masks = masks.to(device)
            labels = torch.flatten(masks)
            for idx in range(21):
                val_targets[idx] = torch.numel(labels[labels==idx])

        for _, masks in test_loader:
            masks = masks.to(device)
            labels = torch.flatten(masks)
            for idx in range(21):
                test_targets[idx] = torch.numel(labels[labels==idx])

        train_targets = train_targets.cpu().numpy()
        val_targets = val_targets.cpu().numpy()
        test_targets = test_targets.cpu().numpy()
        
        figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        xlabels = [str(x) for x in range(21)]

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

    return model, loss_criterion, optimizer, train_loader, val_loader, test_loader, inv_transform

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

    # Move the model to GPU
    model.to(device)
    loss_criterion.to(device)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_criterion, log="all", log_freq=config['log_interval'])

    for epoch_count in range(start_epoch, config['epochs']):
        # Enable training
        model.train()

        # Track the cumulative loss
        running_loss = 0.0
        running_count = 0

        # Track accuracy and IoU
        total_pixels = 0
        total_matches = 0
        intersections = torch.zeros(21, dtype=int).to(device)
        unions = torch.zeros(21, dtype=int).to(device)
        for idx, data in enumerate(train_loader):

            # Get the inputs and labels
            images, masks = data

            # Move the input and output to the GPU
            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)

            # Zero out the gradient of the optimizer
            optimizer.zero_grad()
            
            # Feed the input to the network and get predictions
            predictions = model(images)['out']
            preds = torch.argmax(predictions, dim=1)
    
            # Compute the loss
            loss = loss_criterion(predictions, masks)

            # Perform back propagation
            loss.backward()

            # Take a step towards the minima
            optimizer.step()
           
            # Update training loss
            running_loss += loss.sum().item()
            running_count += config['batch_size']
           
            # Update accuarcy and IoU metrics
            total_pixels += images.numel()
            total_matches = (preds == masks).sum()
            for c in range(21):
                intersections[c] += (preds[preds == masks] == c).sum()
                unions[c] += (preds == c).sum() + (masks == c).sum() - (preds[preds == masks] == c).sum()

        # Log validation and training loss with wandb
        if epoch_count % config['log_interval'] == 0:
            val_loss = 0.0
            val_total_pixels = 0
            val_total_matches = 0
            val_intersections = torch.zeros(21, dtype=int).to(device)
            val_unions = torch.zeros(21, dtype=int).to(device)
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    # Get the input and the label
                    images, masks = data
                    
                    # Move the input and output to the GPU
                    images = images.cuda(non_blocking=True)
                    masks = masks.cuda(non_blocking=True)
                    
                    # Get prediction and compute loss
                    predictions = model(images)['out']
                    preds = torch.argmax(predictions, dim=1)
                   
                    # Compute loss
                    val_loss += loss_criterion(predictions, masks).sum().item()
                    
                    # Update accuarcy and IoU metrics
                    val_total_pixels += images.numel()
                    val_total_matches = (preds == masks).sum()
                    for c in range(21):
                        val_intersections[c] += (preds[preds == masks] == c).sum()
                        val_unions[c] += (preds == c).sum() + (masks == c).sum() - (preds[preds == masks] == c).sum()
        
            # Compute metrics
            val_loss /= len(val_loader.dataset)
            train_loss = running_loss/running_count 
            val_accu = val_total_matches/val_total_pixels
            train_accu = total_matches/total_pixels
            val_iou = (val_intersections / val_unions).sum().item()
            train_iou = (intersections / unions).sum().item() 
            
            # Reset counters
            total_matches = 0
            total_pixels = 0
            running_loss = 0
            running_count = 0
            intersections = torch.zeros(21, dtype=int).to(device)
            unions = torch.zeros(21, dtype=int).to(device)
            
            print(f"[epoch:{epoch_count+1}] Average Training Loss: {train_loss}, Average Validation Loss: {val_loss}")
            
            wandb.log({"Epoch": epoch_count + 1,
                       "Train_loss": train_loss,
                       "Validation_loss": val_loss,
                       "Training accuracy": train_accu,
                       "Validation accuracy": val_accu,
                       "Training IoU": train_iou,
                       "Validation IoU": val_iou,
                       })
    
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

    # Move the model to the GPU
    model.to(device)
    model.eval()

    # Create an array of ground truth labels
    ground_truth = []
    
    # Create an array of predicted labels
    preds = []
    
    # IoU and preds ground truths
    ground_truth_IoU = [[] for x in range(10)]
    preds_IoU = [[] for x in range(10)]

    # Create an array of class names
    class_names = [str(x) for x in range(22)]

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # Get the input and the label
            images, masks = data
            
            # Move the input to the GPU
            images = images.cuda(non_blocking=True)
            
            # Store the ground truth
            ground_truth.extend(list(masks.numpy().flatten()))

            # Get predicted label
            predictions = model(images)
            pred_labels = torch.argmax(predictions, dim=1)
            preds.extend(list(pred_labels.cpu().numpy().flatten()))
            
            # Compute the IoU
            intersections = torch.zeros(21, dtype=int).to(device)
            unions = torch.zeros(21, dtype=int).to(device)
            for c in range(21):
                intersections[c] += (pred_labels[pred_labels == masks] == c).sum()
                unions[c] += (pred_labels == c).sum() + (masks == c).sum() - (pred_labels[pred_labels == masks] == c).sum()
            iou = intersections.sum()/unions.sum()
            for x in range(10):
                iou_range = (x+1)*0.1
                if(iou <= iou_range):
                    ground_truth_IoU[x].extend(list(masks.numpy().flatten()))
                    preds_IoU[x].extend(list(pred_labels.cpu().numpy().flatten()))
                    break

    # Compute accuracy, precision, recall and f1_score
    average_precisions = []
    for x in range(10):
        average_precisions.append(average_precision_score(ground_truth_IoU[x], preds_IoU[x], average='weighted'))
    accuracy = accuracy_score(ground_truth, preds, normalize=True)
    precision = precision_score(ground_truth, preds, average='weighted', zero_division=1)
    recall = recall_score(ground_truth, preds, average='weighted', zero_division=1)
    f_score = f1_score(ground_truth, preds, average='weighted', zero_division=1)

    # Log the metrics
    logs = {"F1_score": f_score,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall
        }
    for x in range(10):
        key = "Average precision for " + str(x*0.1) + " < IoU <= " + str((x+1)*0.1)
        value = average_precisions[x]
        logs[key] = value
    wandb.log(logs)

    # Save the model
    torch.onnx.export(model, input_images, "model_Q1_2.onnx")
    wandb.save("model_Q1_2.onnx")

def analyze_misclassifications(model, test_loader, inv_transform):
    # Move the model to the GPU
    model.to(device)
    model.eval()

    visualization_count = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # Get the input and the label
            images, masks = data
            
            # Move the input to GPU
            images = images.cuda(non_blocking=True)

            # Get predicted label
            predictions = model(images) 
            pred_labels = torch.argmax(predictions, dim=1)

            for idx in range(images.shape[0]):
                image = images.cpu()[idx]
                image = inv_transform(image)
                mask = masks.cpu()[idx]
                pred_label = pred_labels.cpu()[idx]

                # Compute IoU
                intersections = torch.zeros(21, dtype=int).to(device)
                unions = torch.zeros(21, dtype=int).to(device)
                for c in range(21):
                    intersections[c] += (pred_label[pred_label == mask] == c).sum()
                    unions[c] += (pred_label == c).sum() + (mask == c).sum() - (pred_label[pred_label == mask] == c).sum()
                iou = intersections.sum()/unions.sum()
                
                # Save image and mask if iou <= 0.5
                if iou <= 0.5:
                    visualization_count += 1
                    plt.figure()
                    plt.imshow(image)
                    img_name = 'image_' + str(visualization_count)
                    plt.savefig(pictures_path + img_name)
                    plt.figure()
                    plt.imshow(mask)
                    mask_name = 'mask_' + str(visualization_count)
                    plt.savefig(pictures_path + mask_name)

                if visualization_count >= 3:
                    return

def model_pipeline(hyperparameters):

    # Tell wandb to get started
    with wandb.init(project="CV_Assignment01", config=hyperparameters, name="FCN Resnet50"):
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

if __name__ == "__main__":
    model_pipeline(config)
    wandb.finish()
