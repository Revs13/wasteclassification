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

#making the binary labels
binary_labels = {
    'cardboard': 1,
    'glass': 1,
    'metal': 1,
    'paper': 1,
    'plastic': 1,
    'trash': 0
}

#transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

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


# Custom dataset with binary labels
class TrashDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]
        label_name = os.path.basename(os.path.dirname(path))
        label = binary_labels[label_name]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

dataset = TrashDataset(root="data/dataset-resized", transform=None)


#model

epochs = 10

class WasteClassifierLite(nn.Module):
    def __init__(self):
        super(WasteClassifierLite, self).__init__()

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
model = WasteClassifierLite().to(device)

# Loss and optimizer
class_imbalance_ratio = (403 + 501 + 410 + 594 + 482) / 137

pos_weight = torch.tensor([class_imbalance_ratio]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

#assign transforms
train_ds.dataset.transform = train_transform
val_ds.dataset.transform = val_transform

#create loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
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

