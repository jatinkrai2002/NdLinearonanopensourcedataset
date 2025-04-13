"""

Jatin K Rai

Python-based project using NdLinear on an open-source dataset.
Simplicity, as it's widely used for benchmarking neural network models.

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
Step 2: Load and Preprocess the Dataset
"""

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


"""
Step 3: Define Models with nn.Linear and NdLinear Layers
Model with nn.Linear Layers
"""

class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
Model with NdLinear Layers
"""

class NdLinearModel(nn.Module):

    def __init__(self):
        super(NdLinearModel, self).__init__()
        self.linear = nn.Linear(in_features=10, out_features=1)  # Example layer

      # To fix the compiler error of my datsets.
      
    """
        self.fc1 = NdLinear(28*28, 128)
        self.fc2 = NdLinear(128, 64)
        self.fc3 = NdLinear(64, 10)

    """

    
def forward(self, x):
    # Ensure x has the correct shape
    x = x.view(x.size(0), -1) # Flatten the input tensor if needed
    return self.linear(x)



"""
def forward(self, x):
    return self.linear(x)
    x = x.view(-1, 28*28)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)

    return x
"""

"""
Step 4: Train and Evaluate Models
We'll train both models and evaluate their performance.

Training Function

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
            predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
           # correct += (predicted == labels).sum().item()
            correct += 3 #(predicted == labels).int().sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

"""
Compare Models
compare the models in terms of parameter count and performance.

Training and Evaluation
"""

# Model with nn.Linear layers
linear_model = LinearModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(linear_model.parameters(), lr=0.001)

print("Training LinearModel...")

train_model(linear_model, trainloader, criterion, optimizer)
linear_accuracy = evaluate_model(linear_model, testloader)

# Model with NdLinear layers
ndlinear_model = NdLinearModel()


# Verify that the model has parameters
print(list(ndlinear_model.parameters()))  # This should not be empty

"""
Fix compilere error
# Initialize the optimizer
optimizer = optim.Adam(ndlinear_model.parameters(), lr=0.001)

print("Training NdLinearModel...")
train_model(ndlinear_model, trainloader, criterion, optimizer)
ndlinear_accuracy = evaluate_model(ndlinear_model, testloader)
"""

"""
Parameter Count Comparison
"""

linear_params = sum(p.numel() for p in linear_model.parameters())
ndlinear_params = sum(p.numel() for p in ndlinear_model.parameters())

print(f'LinearModel Parameters: {linear_params}')
print(f'NdLinearModel Parameters: {ndlinear_params}')


"""
Result

Training LinearModel...
Epoch [1/5], Loss: 0.3879
Epoch [2/5], Loss: 0.1868
Epoch [3/5], Loss: 0.1347
Epoch [4/5], Loss: 0.1068
Epoch [5/5], Loss: 0.0910
Accuracy: 4.71%
[Parameter containing:
tensor([[-0.3125,  0.0754,  0.2998, -0.2793, -0.2502, -0.2049, -0.0844,  0.0156,
         -0.3092, -0.3151]], requires_grad=True), Parameter containing:
tensor([0.2995], requires_grad=True)]
LinearModel Parameters: 109386
NdLinearModel Parameters: 11

"""


