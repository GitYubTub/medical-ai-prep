import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

data_path = "./data"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
test_data = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch = 64, shuffle=True)
test_loader = DataLoader(test_data, batch = 1000, suffle=True)

class MNIST_predictions(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
           nn.Linear(784,128),
           nn.Relu(),
           nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

model = MNIST_predictions()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters,lr=0.001)
        
def train_epoch(train_loader, optimizer, loss_function, model):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output,target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,predictions = output.max(1) # since outputs.max gives a max value and its index and we only want the index, we use _ to get rid of the first output
        total += target.size(0)
        correct += predictions.eq(target).sum().item() # sums up all the times predictions is equal to the target and coverts from a tensor to python number

        if (i + 1) % 134 == 0:
            avg_loss = running_loss / 134
            accuracy = 100 * correct / total
            print(f"Batch {i + 1}: loss={avg_loss:.4f}, accuracy={accuracy:.2f}%")

            running_loss = 0.0
            correct = 0
            total = 0

    return model

def model_eval(test_loader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for data, target in test_loader:
            output = model(data)
            _,predictions = output.max(1) # since outputs.max gives a max value and its index and we only want the index, we use _ to get rid of the first output
            total += target.size(0)
            correct += predictions.eq(target).sum().item()

        accuracy = 100 * correct / total
        print(f"accuracy={accuracy:.2f}%")

    
num_epochs = 5
test_accuracies = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_epoch(train_loader, optimizer, loss_function, model)
    accuracy = model_eval(test_loader, model)
    test_accuracies.append(accuracy)
