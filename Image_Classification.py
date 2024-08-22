import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision
import time
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import wandb
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from torchvision import transforms


import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

wandb.login()
#05295dd19a5b535a27611bf3695f3b8b9eb13e22

# class_labels = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear':4,
#                 'dog': 5, 'roe_deer': 6, 'sika_deer': 7, 'wild_boar': 8, 'people': 9}

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, root, class_labels, train=True, transform=None):
#         self.root = root
#         self.class_labels = class_labels
#         self.transform = transform

#         # Create a list to hold the image file paths and labels
#         self.data = []
#         self.labels = []

#         # Load data and labels

#         classes = os.listdir(root)
#         for class_folder in classes:
#             folder = os.path.join(root, class_folder)
#             lim = 0
#             for image_file in os.listdir(folder):
#                 if image_file.lower().endswith('.jpg'):
#                     self.data.append(os.path.join(folder, image_file))
#                     self.labels.append(class_labels[class_folder])
#                     #lim += 1
#                     #if lim >= 50:
#                         #break

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image_path = self.data[idx]
#         label = self.labels[idx]

#         # Open the image file
#         image = Image.open(image_path).convert('RGB')

#         # Transform the image if a transform is provided
#         if self.transform:
#             image = self.transform(image)

#         return image, label

#     def set_data(self, data, labels):
#         self.data = data
#         self.labels = labels

class_labels = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear':4,
                'dog': 5, 'roe_deer': 6, 'sika_deer': 7, 'wild_boar': 8, 'people': 9}

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, class_labels, train=True, transform=None):
        self.root = root
        self.class_labels = class_labels
        self.transform = transform

        # Create a list to hold the image file paths and labels
        self.data = []
        self.labels = []

        # Load data and labels

        classes = os.listdir(root)
        for class_folder in classes:
            folder = os.path.join(root, class_folder)
            #lim = 0
            for image_file in os.listdir(folder):
                if image_file.lower().endswith('.jpg'):
                    image = Image.open(os.path.join(folder, image_file)).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                self.data.append(image)
                self.labels.append(class_labels[class_folder])
                    #lim += 1
                    #if lim >= 50:
                        #break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        return image, label

    def set_data(self, data, labels):
        self.data = data
        self.labels = labels



root = 'drive/MyDrive/CV_HW1/Cropped_final'
resize_transformation = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
dataset = CustomDataset(root, class_labels=class_labels, train=True, transform=resize_transformation)

wandb.init(project='cv_hw1_q2', entity='akanksha_singal')

train_size = 0.7
val_size = 0.1
test_size = 0.2

train_idx, test_idx = train_test_split(
    range(len(dataset)),
    test_size=val_size + test_size,
    stratify=dataset.labels,
    random_state=42
)

train_data = torch.utils.data.Subset(dataset, train_idx)
test_val_dataset = torch.utils.data.Subset(dataset, test_idx)

val_size_actual = val_size / (val_size + test_size)
val_idx, test_idx = train_test_split(
    range(len(test_val_dataset)),
    test_size=val_size_actual,
    stratify=[test_val_dataset[i][1] for i in range(len(test_val_dataset))],
    random_state=42
)

val_data = torch.utils.data.Subset(test_val_dataset, val_idx)
test_data = torch.utils.data.Subset(test_val_dataset, test_idx)

BATCH_SIZE = 64
train_data = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True)
val_data = torch.utils.data.DataLoader(val_data, batch_size = BATCH_SIZE, shuffle=False)
test_data = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, shuffle=False)

train_labels = [dataset.labels[idx] for idx in train_idx]

val_labels = [test_val_dataset.dataset.labels[test_val_dataset.indices[i]] for i in val_idx]

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class_labels = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear': 4,
                'dog': 5, 'roe_deer': 6, 'sika_deer': 7, 'wild_boar': 8, 'people': 9}


train_label_counts = Counter(train_labels)
val_label_counts = Counter(val_labels)


sorted_train_counts = [train_label_counts[class_labels[key]] for key in class_labels]
sorted_val_counts = [val_label_counts[class_labels[key]] for key in class_labels]


fig, ax = plt.subplots()

class_names = list(class_labels.keys())
x = np.arange(len(class_names))
width = 0.35

rects1 = ax.bar(x - width/2, sorted_train_counts, width, label='Train')
rects2 = ax.bar(x + width/2, sorted_val_counts, width, label='Validation')

ax.set_ylabel('Counts')
ax.set_title('Data distribution across class labels')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()

"""#Question 2 Part 2"""

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(8192, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

config = wandb.config
config.learning_rate = 0.001

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_data, 0):
        inputs, labels = data[0], data[1]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_data)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})


    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_data:
            images, labels = data[0], data[1]
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_data)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})
    print(f"train_loss: {train_loss}, train_accuracy: {train_accuracy}, val_loss: {val_loss}, val_accuracy: {val_accuracy}")

print('Finished Training')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

"""c. There is some difference between the training and validation loss i.e Training loss is lower than validation loss. We can see these losses are diverging we can say that the model is overfitting."""

model.eval()

true_labels = []
pred_labels = []


with torch.no_grad():
    for data in test_data:
        inputs, labels = data[0], data[1]
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())


true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")

"""d. Accuracy is 52.9% on test dataset
and F1 score is 0.520
"""

conf_mat = confusion_matrix(true_labels, pred_labels)
print(conf_mat)

wandb.log({"test_accuracy": accuracy, "test_f1_score": f1, "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels, preds=pred_labels, class_names=list(class_labels.keys()))})

wandb.finish()

"""#Qestion 2 Part 3"""

wandb.init(project="resnet-finetune", entity="akanksha_singal")

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

class FineTunedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(FineTunedResNet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)

        self.features = nn.Sequential(*list(self.resnet18.children())[:-1])

        num_ftrs = self.resnet18.fc.in_features

        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        classification = self.fc(x)
        return classification, x


config = wandb.config
config.learning_rate = 0.001
model = FineTunedResNet18(num_classes=10)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=config.learning_rate)


num_epochs = 10
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total = 0

    for images, labels in train_data:
        images.size()
        optimizer.zero_grad()

        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_data)
    train_accuracy = correct_preds / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})


    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_data:
            images.size()

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_data)
    val_accuracy = correct_preds / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

model.eval()

true_labels = []
pred_labels = []


with torch.no_grad():
    for data in test_data:
        inputs, labels = data[0], data[1]
        outputs, _ = model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())


true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")

conf_mat = confusion_matrix(true_labels, pred_labels)
print(conf_mat)
b
wandb.log({"test_accuracy": accuracy, "test_f1_score": f1, "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels, preds=pred_labels, class_names=list(class_labels.keys()))})

model.eval()


def extract_features(dataloader, model):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label in dataloader:
            outputs, _ = model(inputs)
            features.append(outputs)
            labels.append(label)
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features.numpy(), labels.numpy()


train_features, train_labels = extract_features(train_data, model)
val_features, val_labels = extract_features(val_data, model)
test_features, test_labels = extract_features(test_data, model)


tsne = TSNE(n_components=2, random_state=123)
train_tsne_features = tsne.fit_transform(train_features)
val_tsne_features = tsne.fit_transform(val_features)

plt.figure(figsize=(10, 8))
for label in set(train_labels):
    indices = train_labels == label
    plt.scatter(train_tsne_features[indices, 0], train_tsne_features[indices, 1], label=label)
plt.legend()
plt.title('t-SNE visualization of the training set')
plt.show()

plt.figure(figsize=(10, 8))
for label in set(val_labels):
    indices = val_labels == label
    plt.scatter(val_tsne_features[indices, 0], val_tsne_features[indices, 1], label=label)
plt.legend()
plt.title('t-SNE visualization of the validation set')
plt.show()


tsne = TSNE(n_components=3, random_state=123)
val_tsne_features = tsne.fit_transform(val_features)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for label in set(val_labels):
    indices = val_labels == label
    ax.scatter(val_tsne_features[indices, 0], val_tsne_features[indices, 1], val_tsne_features[indices, 2], label=label)
plt.legend()
plt.title('t-SNE visualization of the validation set in 3D')
plt.show()

wandb.finish()

wandb.init(project="resnet-data-aug", entity="akanksha_singal")

train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Rotates by +/- 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming 3-channel images
])

aug_dataset = CustomDataset(root, class_labels=class_labels, train=True, transform=train_transforms)

train_size = 0.7
val_size = 0.1
test_size = 0.2

train_idx, test_idx = train_test_split(
    range(len(dataset)),
    test_size=val_size + test_size,
    stratify=dataset.labels,
    random_state=42
)

train_data = torch.utils.data.Subset(aug_dataset, train_idx)
BATCH_SIZE = 64
train_data = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True)

train_labels = [aug_dataset.labels[idx] for idx in train_idx]

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

class FineTunedResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(FineTunedResNet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)

        self.features = nn.Sequential(*list(self.resnet18.children())[:-1])

        num_ftrs = self.resnet18.fc.in_features

        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        classification = self.fc(x)
        return classification, x


config = wandb.config
config.learning_rate = 0.001
model = FineTunedResNet18(num_classes=10)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=config.learning_rate)


num_epochs = 10
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total = 0

    for images, labels in train_data:
        images.size()
        optimizer.zero_grad()

        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_data)
    train_accuracy = correct_preds / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})


    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_data:
            images.size()

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_data)
    val_accuracy = correct_preds / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

model.eval()

true_labels = []
pred_labels = []

with torch.no_grad():
    for data in test_data:
        inputs, labels = data[0], data[1]
        outputs, _ = model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")

conf_mat = confusion_matrix(true_labels, pred_labels)
print(conf_mat)

wandb.log({"test_accuracy": accuracy, "test_f1_score": f1, "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=true_labels, preds=pred_labels, class_names=list(class_labels.keys()))})

wandb.finish()