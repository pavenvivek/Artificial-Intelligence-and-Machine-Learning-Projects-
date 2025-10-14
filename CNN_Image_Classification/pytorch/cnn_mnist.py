import os
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

# data preprocessing

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_cnt = int(.10 * len(x_train))
test_cnt  = int(.10 * len(x_test))

x_train = x_train[:train_cnt]
y_train = y_train[:train_cnt]

x_test = x_test[:test_cnt]
y_test = y_test[:test_cnt]

x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

print (f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")


class mnist_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_data = mnist_dataset(x_train, y_train)
train_dataloader = DataLoader(train_data, batch_size=100)

test_data = mnist_dataset(x_test, y_test)
test_dataloader = DataLoader(test_data, batch_size=1)


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    

# model construction

class NeurelNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv   = nn.Conv2d(1, 1, 3)
        self.maxpl  = nn.MaxPool2d(3)
        self.rlu    = nn.ReLU()

        self.fl     = nn.Flatten()
        self.mlp    = nn.LazyLinear(800)
        self.output = nn.Linear(800, 10)
        #self.bn1     = nn.BatchNorm2d(num_features=1)
        self.bn2     = nn.BatchNorm1d(800)
        self.sig    = nn.Sigmoid()

    def forward(self, x):

        x = self.rlu(self.conv(x))
        x = self.rlu(self.conv(x))
        x = self.maxpl(x)
        x = self.rlu(self.mlp(self.fl(x)))
        x = self.bn2(x)
        x = self.output(x)
        x = self.sig(x)

        return x


model = NeurelNet().to(device)

summary(model, input_size=(1, 28, 28))
        
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn   = nn.CrossEntropyLoss()

# model training

def train(train_dataloader, model, optimizer, loss_fn):

   model.train()

   for batch, (x, y) in enumerate(train_dataloader):

       x, y = x.to(device), y.to(device)

       # fordward pass
       pred = model(x)
       loss = loss_fn(pred, y)

       # backward propagation
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       if batch%10 == 0:
           print (f"Training loss: {loss.item()}")

           
# model testing
def test(test_dataloader, model, loss_fn):

    model.eval()

    with torch.no_grad():
        
        count = 0
        total_loss = 0
        acc = 0
        for x, y in test_dataloader:

            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            total_loss = total_loss + loss.item()
            count = count + 1

            if pred.argmax() == y:
                acc = acc + 1

        print (f"Testing loss: {total_loss/count}, accuracy: {acc/count}")


Epochs = 0
for i in range(0, Epochs):
    print (f"Epoch: {i}")
    train(train_dataloader, model, optimizer, loss_fn)
    test(test_dataloader, model, loss_fn)
    print ("-----------")
        

# model prediction
model.eval()


with torch.no_grad():

    i = 0

    for x, y in test_dataloader:

        if i == 10:
            break

        x, y = x.to(device), y.to(device)

        pred = model(x)

        print (f"pred: {pred.argmax()}, y: {y}")
        i = i + 1

