"""
Visualize kernels applied in CNN.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cnn import Conv_Net

conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))
conv_net.eval()

# Get the weights of the first convolutional layer of the network
first_conv_layer = conv_net.conv1
kernels = first_conv_layer.weight.data.cpu().numpy()

# Question 5: Visualize kernels from first convolutional layer
# Create a plot that is a grid of images, where each image is one kernel from the conv layer.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels,
# the grid might have 4 rows and 8 columns. Finally, normalize the values in the grid to be
# between 0 and 1 before plotting.
kernels_normalized = kernels.copy()
for i in range(kernels_normalized.shape[0]):
    kernel = kernels_normalized[i, 0]
    kernel_min = kernel.min()
    kernel_max = kernel.max()
    if kernel_max - kernel_min > 0:
        kernels_normalized[i, 0] = (kernel - kernel_min) / (kernel_max - kernel_min)

# Note: I'm not too familiar with pyplot so I used Claude to help format and make the plots nice.
num_kernels = kernels.shape[0]
grid_size = int(np.ceil(np.sqrt(num_kernels)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
fig.suptitle('Kernels Learned at First Convolutional Layer', fontsize=16, fontweight='bold')
for i in range(grid_size*grid_size):
    row = i // grid_size
    col = i % grid_size
    if i < num_kernels:
        kernel = kernels_normalized[i, 0] # Get single channel
        axes[row, col].imshow(kernel, cmap='gray')
        axes[row, col].set_title(f'K{i}', fontsize=8)
    else:
        axes[row, col].axis('off')
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])
# Save the grid to a file named 'kernel_grid.png'. Add the saved image to the PDF report you submit.
plt.tight_layout()
plt.savefig('kernel_grid.png', dpi=150, bbox_inches='tight')
plt.close()

# Question 6: Apply kernels to sample image
# Apply the kernel to the provided sample image.
img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0 # Normalize the image
img_tensor = torch.tensor(img).float()
img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
print(f"Image tensor shape: {img_tensor.shape}")

# Apply the kernel to the image
with torch.no_grad():
    output = conv_net.conv1(img_tensor)

# convert output from shape (1, num_channels, output_dim_0, output_dim_1) to (num_channels, 1, output_dim_0, output_dim_1) for plotting.
# If not needed for your implementation, you can remove these lines.
output = output.squeeze(0)

output_np = output.cpu().numpy()
output_normalized = np.zeros_like(output_np)
for i in range(output_np.shape[0]):
    channel = output_np[i]
    channel_min = channel.min()
    channel_max = channel.max()
    if channel_max - channel_min > 0:
        output_normalized[i] = (channel - channel_min) / (channel_max - channel_min)

# Create a plot that is a grid of images, where each image is the result of applying one kernel to the sample image.
# Choose dimensions of the grid appropriately. For example, if the first layer has 32 kernels, the grid might have 4 rows and 8 columns.
# Finally, normalize the values in the grid to be between 0 and 1 before plotting.
# Note: I'm not too familiar with pyplot so I used Claude to help format and make the plots nice.
num_outputs = output_normalized.shape[0]
grid_size = int(np.ceil(np.sqrt(num_outputs)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
fig.suptitle('Feature Maps from First Convolutional Layer', fontsize=16, fontweight='bold')
for i in range(grid_size * grid_size):
    row = i // grid_size
    col = i % grid_size
    if i < num_outputs:
        axes[row, col].imshow(output_normalized[i], cmap='gray')
        axes[row, col].set_title(f'Kernel {i}', fontsize=8)
    else:
        axes[row, col].axis('off')   
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])
# Save the grid to a file named 'image_transform_grid.png'. Add the saved image to the PDF report you submit.
plt.tight_layout()
plt.savefig('image_transform_grid.png', dpi=150, bbox_inches='tight')
plt.close()

# Question 7: Feature map progression through layers
# Create a feature map progression. You can manually specify the forward pass order or programatically track each activation through the forward pass of the CNN.
with torch.no_grad():
    x = img_tensor

    # Conv1, ReLU, Pool
    x1_conv = conv_net.conv1(x)
    x1_relu = F.relu(x1_conv)
    x1_pool = conv_net.pool(x1_relu)

    # Conv2, ReLU, Pool
    x2_conv = conv_net.conv2(x1_pool)
    x2_relu = F.relu(x2_conv)
    x2_pool = conv_net.pool(x2_relu)

    # Conv3, ReLU, Pool
    x3_conv = conv_net.conv3(x2_pool)
    x3_relu = F.relu(x3_conv)
    x3_pool = conv_net.pool(x3_relu)

# Show first channel of each layer
layers = [
    ('Original Image', img_tensor.squeeze().cpu().numpy()),
    ('Conv1 Output', x1_conv[0, 0].cpu().numpy()),
    ('Conv1 + ReLU', x1_relu[0, 0].cpu().numpy()),
    ('Conv1 + ReLU + Pool', x1_pool[0, 0].cpu().numpy()),
    ('Conv2 Output', x2_conv[0, 0].cpu().numpy()),
    ('Conv2 + ReLU', x2_relu[0, 0].cpu().numpy()),
    ('Conv2 + ReLU + Pool', x2_pool[0, 0].cpu().numpy()),
    ('Conv3 Output', x3_conv[0, 0].cpu().numpy()),
    ('Conv3 + ReLU', x3_relu[0, 0].cpu().numpy()),
    ('Conv3 + ReLU + Pool', x3_pool[0, 0].cpu().numpy()),
]

# Note: I'm not too familiar with pyplot so I used Claude to help format and make the plots nice.
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Feature Map Progression Through CNN Layers (First Channel Only)', 
             fontsize=16, fontweight='bold')

for idx, (title, feature_map) in enumerate(layers):
    row = idx // 5
    col = idx % 5
    # Normalize for visualization
    fmap = feature_map.copy()
    fmap_min = fmap.min()
    fmap_max = fmap.max()
    if fmap_max - fmap_min > 0:
        fmap = (fmap - fmap_min) / (fmap_max - fmap_min)
    axes[row, col].imshow(fmap, cmap='viridis')
    axes[row, col].set_title(title, fontsize=10, fontweight='bold')
    axes[row, col].axis('off')
# Save the image as a file named 'feature_progression.png'
plt.tight_layout()
plt.savefig('feature_progression.png', dpi=150, bbox_inches='tight')
plt.close()
