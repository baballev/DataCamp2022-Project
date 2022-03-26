import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim


filtered_categories = ["Beverages", "Sweet snacks", "Dairies", "Cereals and potatoes", "Meats", "Fermented foods",
                       "Fermented milk products",
                       "Groceries", "Meals", "Cereals and their products", "Cheeses", "Sauces", "Spreads",
                       "Confectioneries", "Prepared meats",
                       "Frozen foods", "Breakfasts", "Desserts", "Canned foods", "Seafood", "Cocoa and its products",
                       "Fats", "Condiments",
                       "Fishes", "Breads", "Yogurts", "Cakes", "Biscuits", "Pastas", "Legumes"]
filtered_categories = [s.lower() for s in filtered_categories]
nb_classes = len(filtered_categories)

CLASS_TO_INDEX = {filtered_categories[i]: i for i in range(len(filtered_categories))}
INDEX_TO_CLASS = {i: filtered_categories[i] for i in range(len(filtered_categories))}

IMG_SIZE = (256, 256)
BATCH_SIZE = 4

transforms = T.Compose([T.Resize(IMG_SIZE), T.ToTensor()])

# convolution kernel size
kernel_size = 3
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = 2


def one_hot_encoding(row):
    # Since we are reading a coma separated file for labels, the different labels are splitted with ; rather than ,
    m = len(filtered_categories)
    y = torch.zeros(m, dtype=float)
    file_labels = [int(l) for l in row["labels"].split(';')]
    for i in file_labels:
        y[i] = 1.0
    return y


class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, split_folder_root, transform=None):
        super().__init__()
        self.transforms = transform
        self.img_paths = [os.path.join(split_folder_root, "images", f) for f in
                          os.listdir(os.path.join(split_folder_root, "images"))]
        self.df_label = pd.read_csv(os.path.join(split_folder_root, "labels.csv"))

    def __getitem__(self, idx):
        img = plt.imread(self.img_paths[idx])
        img = Image.fromarray(img)
        row = self.df_label.loc[idx]
        y = one_hot_encoding(row)
        if self.transforms is not None:
            X = self.transforms(img)

        return X, y

    def __len__(self):
        return len(self.img_paths)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = 300

        self.img_rows = IMG_SIZE[0]
        self.img_cols = IMG_SIZE[1]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=nb_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(int(IMG_SIZE[0]/4 * IMG_SIZE[1]/4 * nb_filters), 120)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, nb_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ImageClassifier:
    def __init__(self):
        self.n_epochs = 100
        self.net = CNN()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    def fit(self, data_loader):
        for epoch in range(self.n_epochs):
            for data in data_loader:
                img, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(img)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        return self

    def predict_proba(self, data_loader):
        Y_pred = []
        for data in data_loader:
            img, labels = data
            Y_pred.append((self.net(img)))
        return Y_pred


train_set = MultilabelDataset(os.path.join(os.path.abspath(os.path.curdir), "data", "train"), transform=transforms)
test_set = MultilabelDataset(os.path.join(os.path.abspath(os.path.curdir), "data", "test"), transform=transforms)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)

classifier = ImageClassifier().fit(train_loader)
y_pred = ImageClassifier().predict_proba(test_set)
