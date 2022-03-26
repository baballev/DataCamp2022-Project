import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

filtered_categories = ["Beverages", "Sweet snacks", "Dairies", "Cereals and potatoes", "Meats", "Fermented foods", "Fermented milk products", 
                    "Groceries", "Meals", "Cereals and their products", "Cheeses", "Sauces", "Spreads", "Confectioneries", "Prepared meats", 
                    "Frozen foods", "Breakfasts", "Desserts", "Canned foods", "Seafood", "Cocoa and its products", "Fats", "Condiments", 
                    "Fishes", "Breads", "Yogurts", "Cakes", "Biscuits", "Pastas", "Legumes"]
filtered_categories = [s.lower() for s in filtered_categories]

CLASS_TO_INDEX = {filtered_categories[i]:i for i in range(len(filtered_categories))}
INDEX_TO_CLASS = {i:filtered_categories[i] for i in range(len(filtered_categories))}

BATCH_SIZE = 4
IMG_SIZE = (256,256)


def one_hot_encoding(row):
    # Since we are reading a coma separated file for labels, the different labels are splitted with ; rather than , 
    m = len(filtered_categories)
    y = torch.zeros(m, dtype=float)
    file_labels = [int(l) for l in row["labels"].split(';')]
    for i in file_labels:
        y[i] = 1.0
    return y


class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataload, transform=None):
        super().__init__()
        self.transforms = transform

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


class ImageClassifier():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0")

        model = models.resnext50_32x4d(pretrained=True)
        for param in model.parameters():
                param.requires_grad = False
        # replace the last layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(num_features, 30),
                    nn.Sigmoid()
                )

        model.train()
        # transfer the model to the GPU
        self.model = model.to(self.device)
        
        self.criterion = nn.BCELoss(reduction='sum')
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
        

    def fit(self, data_loader):
        max_epoch_number = 1
        epoch = 0
        iteration = 0

        while True:
            batch_losses = []
            for imgs, targets in data_loader:  # ToDo: make batches of data
                imgs, targets = T.ToTensor()(T.Resize(IMG_SIZE)(Image.fromarray(imgs))).to(self.device), torch.from_numpy(targets).to(self.device)
                imgs = imgs.unsqueeze(0)
                targets = targets.unsqueeze(0)
                if imgs.shape[1] != 3: continue # Filter grayscale images
                self.optimizer.zero_grad()
                model_result = self.model(imgs)
                loss = self.criterion(model_result, targets.type(torch.float))
                batch_size = len(imgs)
                batch_loss_value = loss.item()/batch_size
                loss.backward()
                self.optimizer.step()
                batch_losses.append(batch_loss_value)
                iteration += 1
                break
            loss_value = np.mean(batch_losses)
            epoch += 1
            if max_epoch_number < epoch:
                break

    def predict_proba(self, data_loader):
        self.model.eval()
        for i, data in enumerate(data_loader):
            if i == 0:
                imgs = data
                with torch.no_grad():
                    imgs = T.ToTensor()(T.Resize(IMG_SIZE)(Image.fromarray(imgs))).to(self.device)
                    imgs = imgs.unsqueeze(0)
                    if imgs.shape[1] == 1:
                        tmp = torch.zeros((imgs.shape[0], 3, imgs.shape[2], imgs.shape[3]))
                        tmp[0, 0, :, :] = imgs
                        tmp[0, 1, :, :] = imgs
                        tmp[0, 2, :, :] = imgs
                        imgs = tmp
                    model_result = self.model(imgs).cpu()
                    tot = model_result

            if i != 0:
                imgs = data
                with torch.no_grad():
                    imgs = T.ToTensor()(T.Resize(IMG_SIZE)(Image.fromarray(imgs))).to(self.device)
                    imgs = imgs.unsqueeze(0)
                    if imgs.shape[1] == 1:
                        tmp = torch.zeros((imgs.shape[0], 3, imgs.shape[2], imgs.shape[3]))
                        tmp[0, 0, :, :] = imgs
                        tmp[0, 1, :, :] = imgs
                        tmp[0, 2, :, :] = imgs
                        imgs = tmp
                    model_result = self.model(imgs)
                    tot = torch.cat((tot,model_result), 0)
        return tot
