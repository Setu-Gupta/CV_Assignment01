import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)    # First convolution layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)   # Second convolution layer
        self.fc = nn.Linear(64 * 32 * 32, 10)                                               # A fully connected layer to map 64 * 3 * 3 outputs to 10 outputs
        self.init_weights()                                                                 # Initialize the weights
        self.softmax = nn.Softmax(dim=1)                                                    # A softmax layer to get predictions

    def forward(self, x):
        x = self.conv1(x)                   # Apply the first layer
        x = F.relu(x)                       # Apply the activation function
        x = self.conv2(x)                   # Apply the second layer
        x = F.relu(x)                       # Apply the activation function
        x = torch.flatten(x, start_dim=1)   # Flatten the values to feed the fully connected layer
        x = self.fc(x)                      # Apply the fully connected layer

        return x

    # Predicts probabilities of various classes
    def predict(self, x):
        x = self.forward(x)
        return self.softmax(x)

    def predict_label(self, x):
        probs = self.predict(x)
        return torch.argmax(probs, dim=1)
