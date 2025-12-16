"""
In this file you will write the model definition for a feedforward neural network. 

Please only complete the model definition and do not include any training code.

The model should be a feedforward neural network, that accepts 784 inputs
(each image is 28x28, and is flattened for input to the network)
and the output size is 10. Whether you need to normalize outputs using
softmax depends on your choice of loss function.

PyTorch documentation is available at https://pytorch.org/docs/stable/index.html, 
and will specify whether a given loss funciton requires normalized outputs or not.
"""

import torch.nn as nn
import torch.nn.functional as F

class FF_Net(nn.Module):
    """
    Feedforward Neural Network for Fashion-MNIST classification.
    
    This network uses fully connected layers to classify 28x28 grayscale images
    of clothing items into 10 categories.

    Architecture:
        - Input layer: 784 neurons (flattened 28x28 image)
        - Hidden layer 1: 512 neurons with ReLU activation and dropout
        - Hidden layer 2: 256 neurons with ReLU activation and dropout
        - Hidden layer 3: 128 neurons with ReLU activation and dropout
        - Output layer: 10 neurons (one per class)
    """
    def __init__(self):
        super().__init__()
        # Layers
        self.fc1 = nn.Linear(784, 512) # input layer: 784 (28x28 image) -> 512
        self.fc2 = nn.Linear(512, 256) # hidden layer 1: 512 -> 256
        self.fc3 = nn.Linear(256, 128) # hidden layer 2:  256 -> 128
        self.fc4 = nn.Linear(128, 10) # output layer: 128 -> 10 (clothing classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass of the feedforward neural network.
        
        The input image is first flattened from a 2D grid into a 1D vector,
        then passed through multiple fully connected layers with ReLU activations
        and dropout for regularization. The final layer produces raw logits
        (unnormalized scores) for each of the 10 classes.
        
        Args:
            x (torch.Tensor): Input batch of images with shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10) representing
                        unnormalized class scores. These should be passed to
                        a loss function like CrossEntropyLoss.
        """
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Forward pass with ReLU activation function
        x = F.relu(self.fc1(x)) # input layer
        x = self.dropout(x)
        x = F.relu(self.fc2(x)) # hidden layer 1
        x = self.dropout(x)
        x = F.relu(self.fc3(x)) # hidden layer 2
        x = self.dropout(x)
        x = self.fc4(x) # output layer
        return x
