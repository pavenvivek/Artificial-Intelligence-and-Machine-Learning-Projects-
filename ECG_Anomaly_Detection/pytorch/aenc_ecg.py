import os, sys
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf
import kagglehub
import matplotlib.pyplot as plt

from keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary


# device setup

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

    
# data preprocessing

df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
#print(f"bf df -> {df}")

max_v = df.iloc[:,:-1].max().max()
min_v = df.iloc[:,:-1].min().min()

#min_v = tf.reduce_min(df.iloc[:,:-1]).numpy()
#max_v = tf.reduce_max(df.iloc[:,:-1]).numpy()

print (f"min_val -> {min_v}, max_val -> {max_v}")

# normalize values to be between 0 and 1 (provided using sigmoid for reconstruction)
df = pd.concat([df.iloc[:,:-1].map(lambda x: (x - min_v)/(max_v - min_v)) , df.iloc[:,-1]], axis=1)

#print(f"af df -> {df}")


normal_data = df[df[df.columns[-1]] == 1]
abnormal_data = df[df[df.columns[-1]] == 0]

#print (f"normal_data: {normal_data}")
#print (f"abnormal_data: {abnormal_data}")

data_x, data_y = normal_data.iloc[:,:-1], normal_data.iloc[:,-1]
anm_data_x, anm_data_y = abnormal_data.iloc[:,:-1], abnormal_data.iloc[:,-1]

print(f"data_x -> {data_x}")
print(f"data_y -> {data_y}")


train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1, shuffle=True)
anm_train_x, anm_test_x, anm_train_y, anm_test_y = train_test_split(anm_data_x, anm_data_y, test_size=0.1, shuffle=True)

print (f"train_x shape : {train_x}")
print (f"train_y shape : {train_y}")

class ecg_dataset(Dataset):

    def __init__(self, data, label):
        self.data = torch.tensor(np.array(data), dtype=torch.float)
        self.label = torch.tensor(np.array(label), dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    


train_data = ecg_dataset(train_x, train_y)
train_dload = DataLoader(train_data, batch_size=64)

test_data = ecg_dataset(anm_test_x, anm_test_y)
test_dload = DataLoader(test_data, batch_size=64)


#sys.exit(-1)


# model construction

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(140, 800)
        self.h2 = nn.Linear(800, 500)
        self.out = nn.Linear(500, 100)

        self.rlu = nn.ReLU()
        self.b1  = nn.BatchNorm1d(800)
        self.b2  = nn.BatchNorm1d(500)

    def forward(self, x):

        x = self.rlu(self.h1(x))
        x = self.rlu(self.h2(x))
        x = self.out(x)

        return x


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(100, 500)
        self.h2 = nn.Linear(500, 800)
        self.out = nn.Linear(800, 140)

        self.rlu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.b1  = nn.BatchNorm1d(500)
        self.b2  = nn.BatchNorm1d(800)

    def forward(self, x):

        x = self.rlu(self.h1(x))
        x = self.rlu(self.h2(x))
        x = self.sig(self.out(x))

        return x


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x):

        x = self.enc(x)
        x = self.dec(x)

        return x


model = AutoEncoder().to(device)

summary(model, input_size=(140,))    


# model settings

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.L1Loss()

# model training

def train(dataloader, model, optimizer, loss_fn):

    model.train()

    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, x) # sending original input as label for reconstruction 

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100 == 0:
            print (f"Training Loss: {loss.item()}")


# model testing

def test(dataloader, model, loss_fn):

    model.eval()

    loss_v = 0
    cnt = 0
    for x, y in dataloader:

        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, x)

        loss_v = loss_v + loss.item()
        cnt = cnt + 1

    print (f"Testing Loss: {loss_v/cnt}")
    

# Epoch
Epoch=20
for i in range(0, Epoch):
    print (f"Epoch: {i}")
    train(train_dload, model, optimizer, loss_fn)
    print ("----------")

test(test_dload, model, loss_fn)
    
# model prediction

model.eval()
with torch.no_grad():

    #pred = model(train_data.data)
    #loss_fn = nn.L1Loss(reduction='none')
    #loss = loss_fn(pred, train_data.data)
    loss_v = []

    #print (f"------> len train: {len(train_dload)}")
    #print (f"------> len test: {len(test_dload)}")
    
    for x, y in train_data:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, x)

        loss_v.append(loss.item())

    loss = np.array(loss_v)
    #print (f"Train preds: {pred}")
    print (f"Train loss: {loss}")

    threshold = np.mean(loss) + np.std(loss)
    print("\nThreshold: ", threshold)


    #pred = model(test_data.data)
    #loss = loss_fn(pred, test_data.data)

    loss_v = []

    for x, y in test_data:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, x)

        loss_v.append(loss.item())

    loss = np.array(loss_v)

    
    #print (f"Test preds: {pred}")
    print (f"Test loss: {loss}")

    test_loss = torch.tensor(loss) #.numpy()
    preds = torch.lt(test_loss, threshold)
    #preds = preds.squeeze(-1)
    #print (f"preds: {preds}")
    #print (f"anm_test_y: {anm_test_y}")

    print (f"\nAccuracy: {accuracy_score(anm_test_y, preds)}")


    
