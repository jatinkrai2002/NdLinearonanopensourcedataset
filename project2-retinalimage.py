"""

Automated Retinal Image Analysis for Disease Detection is a compelling use case for NdLinear, especially given the importance of preserving spatial information in medical imaging. Let's outline a project that leverages NdLinear for this purpose.

Project: Automated Retinal Image Analysis for Disease Detection
Objective
To develop a deep learning model using NdLinear layers for automated detection of retinal diseases from fundus images, aiming to improve diagnostic accuracy and efficiency.

Dataset
Retina Image Bank dataset, which contains a variety of retinal images annotated with different diseases 
such as diabetic retinopathy, glaucoma, and age-related macular degeneration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

# Assuming NdLinear is implemented in a module named ndlinear
from ndlinear import NdLinear


"""
Load and Preprocess the Dataset
Load the Retina Image Bank dataset and apply necessary transformations.
"""

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True)

testset = torchvision.datasets.ImageFolder(root='./data/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=3, shuffle=False)


"""
Define Models with nn.Linear and NdLinear Layers
Define two models: one using nn.Linear layers and another using NdLinear layers.

Model with nn.Linear Layers
"""

class LinearRetinaModel(nn.Module):
    def __init__(self):
        super(LinearRetinaModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*56*56, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Assuming 3 classes for simplicity


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        # To fix the compiler error of my datsets.
    
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        #  x = self.fc3(x)
        
        return x

"""

Model with NdLinear Layers
"""


class NdLinearRetinaModel(nn.Module):
    def __init__(self):
        super(NdLinearRetinaModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # To fix the compiler error of my datsets.
      
    """
        #   self.fc1 = NdLinear(64*56*56, 128)
        #   self.fc2 = NdLinear(128, 64)
        #   self.fc3 = NdLinear(64, 3)  # Assuming 3 classes for simplicity
    """

     

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

      # To fix the compiler error of my datsets.
       # x = x.view(-1, 64*56*56)
        x = x.view(x.size(0), -1) # Ensure the batch size is preserved
       # x = torch.relu(self.fc1(x))
       # x = torch.relu(self.fc2(x))
       # x = self.fc3(x)
        return x

"""

Train and Evaluate Models
Train both models and evaluate their performance.
"""


def train_model(model, trainloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}')

"""
Evaluation Function
"""

def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

"""
Compare Models
Compare the models in terms of parameter count and performance.
"""

# Model with nn.Linear layers
linear_retina_model = LinearRetinaModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(linear_retina_model.parameters(), lr=0.001)

print("Training LinearRetinaModel...")
train_model(linear_retina_model, trainloader, criterion, optimizer)
linear_accuracy = evaluate_model(linear_retina_model, testloader)

# Model with NdLinear layers
ndlinear_retina_model = NdLinearRetinaModel()
optimizer = optim.Adam(ndlinear_retina_model.parameters(), lr=0.001)

print("Training NdLinearRetinaModel...")
train_model(ndlinear_retina_model, trainloader, criterion, optimizer)
ndlinear_accuracy = evaluate_model(ndlinear_retina_model, testloader)


"""
Parameter Count Comparison
"""

linear_params = sum(p.numel() for p in linear_retina_model.parameters())
ndlinear_params = sum(p.numel() for p in ndlinear_retina_model.parameters())

print(f'LinearRetinaModel Parameters: {linear_params}')
print(f'NdLinearRetinaModel Parameters: {ndlinear_params}')


"""
Result:
Training LinearRetinaModel...
Epoch [1/5], Loss: 15.0112
Epoch [2/5], Loss: 14.9851
Epoch [3/5], Loss: 14.9811
Epoch [4/5], Loss: 14.9846
Epoch [5/5], Loss: 14.9814
Accuracy: 0.00%
Training NdLinearRetinaModel...
Epoch [1/5], Loss: 14.9021
Epoch [2/5], Loss: 14.5705
Epoch [3/5], Loss: 14.2263
Epoch [4/5], Loss: 13.8559
Epoch [5/5], Loss: 13.4176
Accuracy: 16.67%
LinearRetinaModel Parameters: 25718083
NdLinearRetinaModel Parameters: 19392

"""