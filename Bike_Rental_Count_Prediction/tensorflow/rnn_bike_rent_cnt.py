import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

#from sklearn.datasets import fetch_california_housing
from keras import layers
import kagglehub

path = kagglehub.dataset_download("marklvl/bike-sharing-dataset")

print("Path to dataset files:", path)

data = tf.keras.utils.get_file("hour.csv", origin=f"file://{path}/hour.csv")

df = pd.read_csv(data)
print (f"df -> {df[df['dteday'] == '2011-01-01']}")


def get_features(df, timesteps):

    x_lst = []
    y_lst = []
    i = 0

    while i+timesteps < len(df):

        #print (f"val -> {train_x.loc[i:i+23, ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']]}")
        #print (f"val -> {np.array(train_x.loc[i:i+23, ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']])}")

        x, y = df.iloc[i:i+timesteps, :-1], df.iloc[i+timesteps-1, -1]
        
        x = np.array(x.loc[:, ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']])
        #x = np.pad(x, pad_width=((0,timesteps-len(x)), (0,0)))
        x_lst.append(x)

        y = np.array(y)
        #y = np.pad(y, pad_width=(0,timesteps-len(y)))
        y_lst.append(y)

        i = i + 1

        #if i > 50:
        #    break

    return np.array(x_lst), np.array(y_lst)


timesteps = 10
x, y = get_features(df,timesteps)
#test_x, test_y = get_features(test_x, test_y)
                                 
print (f"x -> {x[0]},\ny -> {y[0]}")

train_cnt = int(.90 * len(df))

train_x, train_y = x[:train_cnt], y[:train_cnt]
test_x, test_y = x[train_cnt:], y[train_cnt:]

                                 
# model construction


inputs  = layers.Input(shape=(timesteps, 13))
rnn1    = layers.SimpleRNN(10)(inputs)
#rnn1    = layers.LSTM(10)(inputs)
#rnn1    = layers.GRU(10, return_sequences=True)(inputs)
#rnn2    = layers.GRU(10)(rnn1)
h1      = layers.Dense(200, activation="relu")(rnn1)
outputs = layers.Dense(1)(h1)

model = keras.Model(inputs=inputs, outputs=outputs, name="RNN")

model.summary()

# model settings

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss="mean_squared_error")

# model training

model.fit(train_x, train_y,
          epochs=20,
          batch_size=64)

# model testing

loss = model.evaluate(test_x, test_y)

print (f"Total loss: {loss}")

# model prediction

for i in range(0,10):

    pred = model.predict(np.array([test_x[i]]))

    print (f"pred: {pred}, y: {test_y[i]}")
