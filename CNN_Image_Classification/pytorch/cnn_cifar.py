import os, sys
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from keras import layers
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# data preprocessing

#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Example normalization
    ])

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_dload = DataLoader(trainset, batch_size=100)
test_dload  = DataLoader(testset, batch_size=1)

#print (f"trainset: {trainset[0]}")
train_x, train_y = trainset[0]
print (f"train_x shape: {train_x.shape}, train_y: {train_y}")
#sys.exit(-1)

'''
(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()

train_cnt = int(1 * len(train_x))
test_cnt  = int(1 * len(test_x))

train_x = train_x[:train_cnt]
train_y = train_y[:train_cnt]

test_x = test_x[:test_cnt]
test_y = test_y[:test_cnt]

print (f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
#print (f"train_x: {train_x[0].shape}")

class cifar_dataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float)
        self.label = torch.tensor(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    

train_data  = cifar_dataset(train_x, train_y)
train_dload = DataLoader(train_data, batch_size=64)

test_data   = cifar_dataset(test_x, test_y)
test_dload  = DataLoader(test_data, batch_size=1)
'''

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

# model construction

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1  = nn.Conv2d(3, 32, 3)
        self.conv2  = nn.Conv2d(32, 64, 3)
        self.conv3  = nn.Conv2d(64, 64, 3)
        self.maxpl1 = nn.MaxPool2d(3)
        self.maxpl2 = nn.MaxPool2d(2)
        self.flt    = nn.Flatten()
        self.h1     = nn.LazyLinear(500)
        self.bn0    = nn.BatchNorm2d(num_features=32)
        self.bn1    = nn.BatchNorm2d(num_features=64)
        self.bn2    = nn.BatchNorm1d(500)
        self.output = nn.Linear(500, 100)
        
        self.rlu  = nn.ReLU()
        #self.sfm = nn.Softmax(dim=1)

    def forward(self, x):

        '''
        x = self.bn0(self.rlu(self.conv1(x)))
        x = self.bn1(self.rlu(self.conv2(self.maxpl1(x))))
        x = self.bn1(self.rlu(self.conv3(self.maxpl2(x))))
        x = self.bn2(self.rlu(self.h1(self.flt(x))))
        x = self.output(x)
        
        '''
        x = self.rlu(self.conv1(x))
        x = self.rlu(self.conv2(self.maxpl1(x)))
        x = self.rlu(self.conv3(self.maxpl2(x)))
        x = self.bn2(self.rlu(self.h1(self.flt(x))))
        x = self.output(x)
        
        return x


model = NeuralNet().to(device)

summary(model, input_size=(3, 32, 32))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn   = nn.CrossEntropyLoss()


# model training

def train(train_dload, model, optimizer, loss_fn):

    model.train()

    for batch, (x,y) in enumerate(train_dload):

        x, y = x.to(device), y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100 == 0:
            print (f"Training loss: {loss.item()}")


# model testing

def test(test_dload, model, loss_fn):

    model.eval()

    count  = 0
    acc    = 0
    loss_v = 0
    for x, y in test_dload:

        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss_v = loss_v + loss.item()
        count = count + 1

        if pred.argmax() == y:
            acc = acc + 1

    print (f"Testing loss: {loss_v/count}, Accuracy: {acc/count}")


# model run    
    
Epoch=30
for i in range(0, Epoch):
    print (f"Epoch: {i}")
    train(train_dload, model, optimizer, loss_fn)
    test(test_dload, model, loss_fn)
    print (f"----------")

# prediction


model.eval()
with torch.no_grad():

    i = 0
    for x, y in test_dload:

        if i == 10:
            break
        
        x, y = x.to(device), y.to(device)
        pred = model(x)

        print (f"pred: {pred.argmax()}, y: {y}")
        i = i + 1


