import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from collections import Counter

#transforms
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class TrashDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)
        self.class_map = {
            'cardboard': 1,
            'glass': 1,
            'metal': 1,
            'paper': 1,
            'plastic': 1,
            'trash': 0
        }
        self.custom_transform = transform
        self.loader = datasets.folder.default_loader

    def __getitem__(self, index):
        path, _ = self.samples[index]
        label_name = os.path.basename(os.path.dirname(path))
        label = self.class_map[label_name]
        img = self.loader(path)
        if self.custom_transform:
            img = self.custom_transform(img)
        return img, label

train_dataset = TrashDataset(root="data/dataset-resized", transform=train_transform)
val_dataset   = TrashDataset(root="data/dataset-resized", transform=val_transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_indices, val_indices = torch.utils.data.random_split(range(len(train_dataset)), [train_size, val_size])

from torch.utils.data import Subset

train_ds = Subset(train_dataset, train_indices)
val_ds = Subset(val_dataset, val_indices)


#model

epochs = 10

class WasteClassifier(nn.Module):
    def __init__(self):
        super(WasteClassifier, self).__init__()

        #the input here is of shape [batch size, 3, 64, 64] because
        #picture means RGB values means 3 features, and it's 64 x 64
        #out_channels=8 means the computer will come up with 8 features for the images

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            # Output: [16, 64, 64]

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # Output: [16, 32, 32]

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), 
            # Output: [32, 32, 32]

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: [32, 16, 16]
        )

        self.fc_layer = nn.Sequential(nn.Linear(32 * 16 * 16, 100), nn.ReLU(), nn.Linear(100, 1))
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WasteClassifier().to(device)

# Loss and optimizer
class_imbalance_ratio = (403 + 501 + 410 + 594 + 482) / 137

pos_weight = torch.tensor([class_imbalance_ratio]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_labels = [train_dataset[i][1] for i in train_indices]
class_counts = Counter(train_labels)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

sample_weights = [class_weights[label] for label in train_labels]
sample_weights = torch.DoubleTensor(sample_weights)
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=32)


# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).squeeze(1).long()
        correct += (predicted == labels.long()).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


#-----------------------
from sklearn.metrics import confusion_matrix, classification_report

all_preds = []
all_labels = []

model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).squeeze(1).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=["Non-Recyclable", "Recyclable"]))
