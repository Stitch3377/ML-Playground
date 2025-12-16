"""
In this file you will write the model definition for a convolutional neural network.
Please only complete the model definition and do not include any training code.

The model should be a convolutional neural network, that accepts 28x28 grayscale images as input, 
and outputs a tensor of size 10.
The number of layers/kernels, kernel sizes and strides are up to you. 

Please refer to the following for more information about convolutions, pooling, 
and convolutional layers in PyTorch:
    - https://deeplizard.com/learn/video/YRhxdVk_sIs
    - https://deeplizard.com/resource/pavq7noze2
    - https://deeplizard.com/resource/pavq7noze3
    - https://setosa.io/ev/image-kernels/
    - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

Whether you need to normalize outputs using softmax depends on your choice of loss function.
PyTorch documentation is available at https://pytorch.org/docs/stable/index.html,
and will specify whether a given loss funciton requires normalized outputs or not.
"""

import torch.nn as nn
import torch.nn.functional as F

class Conv_Net(nn.Module):
    """
    Convolutional Neural Network for Fashion-MNIST classification.
    
    The network progressively increases channel depth (1 -> 32 -> 64 -> 128)
    while reducing spatial dimensions (28x28 -> 14x14 -> 7x7 -> 3x3) through
    max pooling. This hierarchical feature extraction allows the network to
    learn patterns from simple edges to complex clothing features.

    Architecture:
        - 3 convolutional blocks (Conv2d -> ReLU -> MaxPool2d)
        - 3 fully connected layers
        - Dropout for regularization
    """
    def __init__(self):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # 1 -> 32 channels, 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32 -> 64 channels, 14x14 -> 14x14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 -> 128 channels, 7x7 -> 7x7
        self.pool = nn.MaxPool2d(2, 2) # pooling layer

        # Fully connected layers
        self.fc1 = nn.Linear(128*3*3, 256) # input layer: 128 channels * 3 * 3 spatial dims
        self.fc2 = nn.Linear(256, 128) # hidden layer: 256 -> 128
        self.fc3 = nn.Linear(128, 10) # output layer: 128 -> 10 (clothing classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass through the convolutional neural network.
        
        Args:
            x (torch.Tensor): Input batch of images with shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10) representing
                        unnormalized class scores. These should be passed to
                        a loss function like CrossEntropyLoss.
        """
        # First conv block
        x = self.conv1(x) # (batch, 1, 28, 28) -> (batch, 32, 28, 28)
        x = F.relu(x)
        x = self.pool(x) # (batch, 32, 28, 28) -> (batch, 32, 14, 14)

        # Second conv block
        x = self.conv2(x) # (batch, 32, 14, 14) -> (batch, 64, 14, 14)
        x = F.relu(x)
        x = self.pool(x) # (batch, 64, 14, 14) -> (batch, 64, 7, 7)

        # Third conv block
        x = self.conv3(x) # (batch, 64, 7, 7) -> (batch, 128, 7, 7)
        x = F.relu(x)
        x = self.pool(x) # (batch, 128, 7, 7) -> (batch, 128, 3, 3)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1) # (batch, 128, 3, 3) -> (batch, 128*3*3)

        # Foward pass with ReLU activation function
        x = F.relu(self.fc1(x)) # input layer
        x = self.dropout(x)
        x = F.relu(self.fc2(x)) # hidden layer
        x = self.dropout(x)
        x = self.fc3(x) # output layer

        return x
