import os, sys
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

#from sklearn.datasets import fetch_california_housing
from keras import layers
import kagglehub

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

        #if i > 50:
        #    break

    return np.array(x_lst), np.array(y_lst)


timesteps = 10
x, y = get_features(df,timesteps)
                                 
print (f"x -> {x[0]},\ny -> {y[0]}")


train_cnt = int(.90 * len(df))

train_x, train_y = x[:train_cnt], y[:train_cnt]
test_x, test_y = x[train_cnt:], y[train_cnt:]

print (f"train: x -> {test_x[0]},\ny -> {test_y[0]}")

#sys.exit(-1)


# model construction


inputs = layers.Input(shape=(timesteps, 1))
rnn    = layers.SimpleRNN(100)(inputs)
#rnn    = layers.GRU(100)(inputs)
h1     = layers.Dense(500, activation="relu")(rnn)
outputs = layers.Dense(1)(h1)

model = keras.Model(inputs=inputs, outputs=outputs, name="rnn")
model.summary()

# model settings

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss="mean_squared_error")

# model training

model.fit(train_x, train_y,
          epochs=40,
          batch_size=64)

# model testing

loss = model.evaluate(test_x, test_y)

print (f"Loss: {loss}")

# model prediction

for i in range(0, 50):

    #print (f"test_x[i] -> {test_x[i]}")
    
    pred = model.predict(np.array([test_x[i]]))

    print (f"pred: {pred}, y: {test_y[i]}")
