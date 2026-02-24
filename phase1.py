import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
from torchvision.datasets import ImageFolder

# Transforms

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(
        size=224,
        scale=(0.9, 1.0),
        ratio=(0.95, 1.05)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    #transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Dataset & DataLoaders

dataset_dir = "/home/hanieh/Downloads/images_phase1"

train_ds = ImageFolder(
    root=os.path.join(dataset_dir, "train"),
    transform=train_transform
)

val_ds = ImageFolder(
    root=os.path.join(dataset_dir, "val"),
    transform=val_transform
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

num_classes = len(train_ds.classes)
print("Classes:", train_ds.classes)

# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model

model = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last residual block
for param in model.layer2.parameters():
    param.requires_grad = True

for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

# Unfreeze classifier only
for param in model.fc.parameters():
    param.requires_grad = True

def freeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

#freeze_batchnorm(model)

# Loss & Optimizer (NO class weights)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# optimizer = torch.optim.Adam([
#     {"params": model.layer4.parameters(), "lr": 1e-4},   # higher is safe
#     {"params": model.fc.parameters(),     "lr": 1e-3}
# ], weight_decay=1e-4)

optimizer = torch.optim.Adam([
    {"params": model.layer2.parameters(), "lr": 1e-5},
    {"params": model.layer3.parameters(), "lr": 5e-5},
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(),     "lr": 5e-4},
], weight_decay=5e-4)

# Train / Eval functions

def train_one_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(1) == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total

# Scheduler

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=5,
    min_lr=1e-6
)

# Early-Stopping

early_stop_patience = 3
epochs_no_improve = 0
best_val_loss = float("inf")

# Training Loop

epochs = 20

train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_loss = float("inf")

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stop_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"LR {optimizer.param_groups[0]['lr']:.2e} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_acc:.4f}"
    )

import matplotlib.pyplot as plt

epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(14, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.axvline(
    val_losses.index(min(val_losses)) + 1,
    linestyle="--",
    label="Best Epoch"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label="Train Accuracy")
plt.plot(epochs_range, val_accs, label="Val Accuracy")
plt.axvline(
    val_losses.index(min(val_losses)) + 1,
    linestyle="--",
    label="Best Epoch"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Test

test_ds = ImageFolder(
    root=os.path.join(dataset_dir, "test"),
    transform=val_transform   # IMPORTANT: no augmentation
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

class_names = test_ds.classes
print("Test classes:", class_names)

# Load the Bset Model

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Compute Test Accuracy + Collect Predictions

import numpy as np

all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Build Confusion Matrix

from sklearn.metrics import confusion_matrix

y_true = np.concatenate(all_labels)
y_pred = np.concatenate(all_preds)

cm = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix

import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()












