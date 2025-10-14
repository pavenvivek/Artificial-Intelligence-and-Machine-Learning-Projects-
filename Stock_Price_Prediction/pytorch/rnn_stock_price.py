import os, sys
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import kagglehub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

path = kagglehub.dataset_download("olegshpagin/mastercard-stock-price-prediction-dataset")

print("Path to dataset files:", path)
#sys.exit(-1)

data = tf.keras.utils.get_file("MA.US_D1.csv", origin=f"file://{path}/D1/MA.US_D1.csv")

df = pd.read_csv(data, sep=",")

print (f"df -> {df}")

#sys.exit(-1)
df = df.loc[:, ['low']] #['open', 'high', 'low', 'close']]
print (f"df -> {df}")

def get_features(df, timesteps):

    x_lst = []
    y_lst = []
    i = 0

    while i+timesteps < len(df):

        #x, y = df.iloc[i:i+timesteps, :-1], df.iloc[i+timesteps-1, -1]
        x, y = df.iloc[i:i+timesteps], df.iloc[i+timesteps]
        
        x = np.array(x)
        x_lst.append(x)

        y = np.array(y)
        y_lst.append(y)

        i = i + 1

    return np.array(x_lst), np.array(y_lst)


timesteps = 10
x, y = get_features(df,timesteps)
                                 
print (f"x -> {x[0]},\ny -> {y[0]}")


train_cnt = int(.90 * len(df))

train_x, train_y = x[:train_cnt], y[:train_cnt]
test_x, test_y = x[train_cnt:], y[train_cnt:]

print (f"train: x -> {test_x[0]},\ny -> {test_y[0]}")

#sys.exit(-1)


class stock_dataset(Dataset):

    def __init__(self, data, label):
        self.data  = torch.tensor(data, dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


train_data = stock_dataset(train_x, train_y)
train_dload = DataLoader(train_data, batch_size=64)

test_data = stock_dataset(test_x, test_y)
test_dload = DataLoader(test_data, batch_size=1)

# model construction

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(1, 10, batch_first=True)
        #self.rnn = nn.GRU(1, 10, batch_first=True)
        self.h1  = nn.Linear(10, 200)
        self.output = nn.Linear(200, 1)

        self.rlu = nn.ReLU()
        #self.bn  = nn.BatchNorm1d(800)

    def forward(self, x):

        h0 = torch.zeros(1, x.size(0), 10)
        x, _ = self.rnn(x, h0)
        x = self.rlu(self.h1(x[:,-1,:]))
        x = self.output(x)

        return x


model = NeuralNet().to(device)

summary(model, input_size=(timesteps, 1))

# model settings

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# model training

def train(model, dataloader, optimizer, loss_fn):

    model.train()

    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%10 == 0:
            print(f"Training Loss: {loss.item()}")


# model testing

def test(model, dataloader, loss_fn):

    model.eval()

    loss_v = 0
    cnt    = 0
    for x, y in dataloader:

        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss_v = loss_v + loss.item()
        cnt = cnt + 1

    print (f"Testing Loss: {loss_v/cnt}")


# Epochs
Epochs=40
for i in range(0, Epochs):
    print (f"Epoch: {i}")
    train(model, train_dload, optimizer, loss_fn)
    print ("----------")

print ("\n")
test(model, test_dload, loss_fn)
print ("\n")
    
# model prediction

model.eval()
with torch.no_grad():

    i = 0
    for x, y in test_dload:

        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        print (f"pred: {pred}, y: {y}")

        if i > 40:
            break
        
        i = i + 1
