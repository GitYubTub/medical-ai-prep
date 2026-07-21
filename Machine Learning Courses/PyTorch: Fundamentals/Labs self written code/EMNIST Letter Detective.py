import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

data_path = "./EMNIST_data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1736,),(0.3317,))
])

train_data = datasets.EMNIST(
    root=data_path,
    split="letters",
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.EMNIST(
    root=data_path,
    split="letters",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

class EMNIST_Predictions(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,26)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

model = EMNIST_Predictions()   
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(train_loader, model, loss_function, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (data, target) in enumerate(train_loader):
        target = target - 1
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = output.max(1)
        correct += predictions.eq(target).sum().item()
        total += target.size(0)

        if (i+1) % 134 == 0:
            avg_loss = running_loss / 134
            accuracy = 100 * correct / total
            print(f"Batch {i+1}: loss={avg_loss:.4f}, accuracy={accuracy:.2f}%")

            running_loss = 0.0
            correct = 0
            total = 0

    return model

def model_eval(test_loader, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

    for data, target in (test_loader):
        target = target - 1
        output = model(data)
        _,predictions = output.max(1)
        correct += predictions.eq(target).sum().item()
        total += target.size(0)

    accuracy = 100 * correct / total
    print(f"accuracy={accuracy:.2f}%")

num_epochs = 5
test_accuracies = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_epoch(train_loader, model, loss_function, optimizer)
    accuracy = model_eval(test_loader, model)
    test_accuracies.append(accuracy)