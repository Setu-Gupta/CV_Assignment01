import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)    # First convolution layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)   # Second convolution layer
        self.fc = nn.Linear(64 * 3 * 3, 10)                                                 # The classification head

    def forward(self, x):
        x = self.conv1(x)   # Apply the first layer
        x = F.relu(x)       # Apply the activation function
        x = self.conv2(x)   # Apply the second layer
        x = F.relu(x)       # Apply the activation function
        x = torch.flatten(x)# Flatten the values to feed the fully connected layer
        x = self.fc(x)      # Apply the classification head
        return x
