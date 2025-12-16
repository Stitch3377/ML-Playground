"""
In this file you will write end-to-end code to train two neural networks to categorize
fashion-mnist data, one with a feedforward architecture and the other with a
convolutional architecture. You will also write code to evaluate the models and generate plots.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from cnn import Conv_Net
from ffn import FF_Net

'''
Configuration: Set which models to train
'''
TRAIN_FFN = True   # Set to True to train FFN
TRAIN_CNN = True  # Set to True to train CNN

'''

PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted.
Please do not change the transforms given below - the autograder assumes these.

'''

transform = transforms.Compose([ # Use transforms to convert images to tensors and normalize them
    transforms.ToTensor(), # convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]) # Common method for grayscale images
])

batch_size = 64

'''

PART 2:
Load the dataset. Make sure to utilize the transform and batch_size from the last section.

'''

trainset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)
testset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

'''

PART 3:
Complete the model defintion classes in ffn.py and cnn.py. We instantiate the models below.

'''

feedforward_net = FF_Net()
conv_net = Conv_Net()

'''

PART 4:
Choose a good loss function and optimizer - you can use the same loss for both networks.

'''

criterion = nn.CrossEntropyLoss()
optimizer_ffn = optim.Adam(feedforward_net.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.001)

'''

PART 5:
Train both your models, one at a time!
(You can train them simultaneously if you have a powerful enough computer,
and are using the same number of epochs,but it is not recommended for this assignment.)

'''

# Train FFN if flag is True
if TRAIN_FFN:
    ffn_losses = []
    ffn_avg_loss = 0
    num_epochs_ffn = 10
    for epoch in range(num_epochs_ffn):  # loop over the dataset multiple times
        running_loss_ffn = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Flatten inputs for ffn (batch_size, 1, 28, 28) -> (batch_size, 784)
            inputs = inputs.view(inputs.size(0), -1)

            # zero the parameter gradients
            optimizer_ffn.zero_grad()

            # forward + backward + optimize
            outputs = feedforward_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ffn.step()
            running_loss_ffn += loss.item()
        ffn_avg_loss = running_loss_ffn / len(trainloader)
        ffn_losses.append(ffn_avg_loss)
        print(f"Epoch {epoch+1} Training Loss: {running_loss_ffn / len(trainloader):.4f}")
        print(f"Total Training Loss: {running_loss_ffn}")

    print('Finished Training FFN')
    torch.save(feedforward_net.state_dict(), 'ffn.pth')  # Saves model file (upload with submission)
else:
    print('Loading saved FFN model')
    feedforward_net.load_state_dict(torch.load('ffn.pth'))

# Train CNN if flag is True
if TRAIN_CNN:
    cnn_losses = []
    cnn_avg_loss = 0
    num_epochs_cnn = 10
    for epoch in range(num_epochs_cnn):  # loop over the dataset multiple times
        running_loss_cnn = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer_cnn.zero_grad()

            # forward + backward + optimize
            outputs = conv_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_cnn.step()
            running_loss_cnn += loss.item()
        cnn_avg_loss = running_loss_cnn / len(trainloader)
        cnn_losses.append(cnn_avg_loss)
        print(f"Epoch {epoch+1} Training Loss: {running_loss_cnn/len(trainloader):.4f}")
        print(f"Total Training Loss: {running_loss_cnn}")

    print('Finished Training CNN')
    torch.save(conv_net.state_dict(), 'cnn.pth')  # Saves model file (upload with submission)
else:
    print('Loading saved CNN model')
    conv_net.load_state_dict(torch.load('cnn.pth'))

'''

PART 6:
Evalute your models! Accuracy should be greater or equal to 80% for both models.
Code to load saved weights commented out below - may be useful for debugging.

'''

# feedforward_net.load_state_dict(torch.load('ffn.pth'))
# conv_net.load_state_dict(torch.load('cnn.pth'))

correct_ffn = 0
total_ffn = 0

correct_cnn = 0
total_cnn = 0

# Since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data


        # Evaluate FFN
        if TRAIN_FFN:
            images_flat = images.view(images.size(0), -1) # Flatten images for FFN
            outputs_ffn = feedforward_net(images_flat)
            _, predicted_ffn = torch.max(outputs_ffn, 1)
            total_ffn += labels.size(0)
            correct_ffn += (predicted_ffn == labels).sum().item()

        # Evaluate CNN
        if TRAIN_CNN:
            outputs_cnn = conv_net(images)
            _, predicted_cnn = torch.max(outputs_cnn, 1)
            total_cnn += labels.size(0)
            correct_cnn += (predicted_cnn == labels).sum().item()

if total_ffn > 0:
    print('Accuracy for feedforward network: ', correct_ffn/total_ffn)
if total_cnn > 0:
    print('Accuracy for convolutional network: ', correct_cnn/total_cnn)

'''

PART 7:
Check the instructions PDF. You need to generate some plots. 

'''

feedforward_net.eval()
conv_net.eval()

classes = ["T-shirt/top", "Trouser",
    "Pullover", "Dress",
    "Coat", "Sandal",
    "Shirt", "Sneaker",
    "Bag", "Ankle boot"
]

# Question 1: Find Correctly and Incorrectly Classified Images
def find_prediction_examples(model, is_ffn=True):
    """Find one correct and one incorrect prediction"""
    correct_example = None
    incorrect_example = None

    with torch.no_grad():
        for images, labels in testloader:
            if is_ffn:
                images_input = images.view(images.size(0), -1)
            else:
                images_input = images

            outputs = model(images_input)
            _, predicted = torch.max(outputs, 1)

            for i, p in enumerate(predicted):
                if correct_example is None and p == labels[i]:
                    correct_example = (images[i], labels[i].item(), p.item())
                if incorrect_example is None and p != labels[i]:
                    incorrect_example = (images[i], labels[i].item(), p.item())
                if correct_example and incorrect_example:
                    return correct_example, incorrect_example
    return correct_example, incorrect_example

# Get examples
ffn_correct, ffn_incorrect = find_prediction_examples(feedforward_net, is_ffn=True)
cnn_correct, cnn_incorrect = find_prediction_examples(conv_net, is_ffn=False)

# Note: I'm not too familiar with pyplot so I used Claude to help format and make the plots nice.
# Plot FFN Examples
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Feedforward Network Predictions', fontsize=14, fontweight='bold')
img = ffn_correct[0].squeeze().numpy() # Correct prediction
axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'CORRECT\nTrue: {classes[ffn_correct[1]]}\nPredicted: {classes[ffn_correct[2]]}')
axes[0].axis('off')
img = ffn_incorrect[0].squeeze().numpy() # Incorrect prediction
axes[1].imshow(img, cmap='gray')
axes[1].set_title(f'INCORRECT\nTrue: {classes[ffn_incorrect[1]]}\nPredicted: {classes[ffn_incorrect[2]]}', color='red')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('ffn_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot CNN Examples
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Convolutional Network Predictions', fontsize=14, fontweight='bold')
img = cnn_correct[0].squeeze().numpy() # Correct prediction
axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'CORRECT\nTrue: {classes[cnn_correct[1]]}\nPredicted: {classes[cnn_correct[2]]}')
axes[0].axis('off')
img = cnn_incorrect[0].squeeze().numpy() # Incorrect prediction
axes[1].imshow(img, cmap='gray')
axes[1].set_title(f'INCORRECT\nTrue: {classes[cnn_incorrect[1]]}\nPredicted: {classes[cnn_incorrect[2]]}', color='red')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('cnn_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

# Question 2: Training Loss Over Time
if 'ffn_losses' in locals() and len(ffn_losses) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(ffn_losses) + 1), ffn_losses, marker='o', label='FFN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Feedforward Network Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ffn_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

if 'cnn_losses' in locals() and len(cnn_losses) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cnn_losses) + 1),
             cnn_losses,
             marker='o',
             label='CNN Training Loss',
             color='orange'
        )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convolutional Network Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cnn_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

'''

PART 8:
Compare the performance and characteristics of FFN and CNN models.

'''

# Question 3: Count Parameters
def count_parameters(model):
    """Counts total parameters in the provided model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

ffn_params = count_parameters(feedforward_net)
cnn_params = count_parameters(conv_net)

print("\nModel Parameters:")
print(f"FNN Total Parameters: {ffn_params:,}")
print(f"CNN Total Parameters: {cnn_params:,}")

with open('model_parameters.txt', 'w', encoding='utf-8') as f:
    f.write(f"FFN Total Parameters: {ffn_params:,}\n")
    f.write(f"CNN Total Parameters: {cnn_params:,}\n")

# Question 4: Confusion Matrices
def generate_confusion_matrix(model, is_ffn=True):
    """Generate confusion matrix for a provided model"""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            if is_ffn:
                images_input = images.view(images.size(0), -1)
            else:
                images_input = images
            outputs = model(images_input)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return confusion_matrix(all_labels, all_preds)

# Generate confusion matrices
ffn_cm = generate_confusion_matrix(feedforward_net, is_ffn=True)
cnn_cm = generate_confusion_matrix(conv_net, is_ffn=False)

# Note: I'm not too familiar with pyplot so I used Claude to help format and make the plots nice.
# Plot FFN
plt.figure(figsize=(10, 8))
sns.heatmap(ffn_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
plt.title('Feedforward Network Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('ffn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot CNN
plt.figure(figsize=(10, 8))
sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
plt.title('Convolutional Network Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('cnn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
