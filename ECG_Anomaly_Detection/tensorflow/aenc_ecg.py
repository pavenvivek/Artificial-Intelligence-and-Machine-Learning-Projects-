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


'''
plt.grid()
plt.plot(np.arange(140), train_x[0])
plt.title("A Normal ECG")
plt.show()

plt.grid()
plt.plot(np.arange(140), anm_train_x[0])
plt.title("An Anomalous ECG")
plt.show()

sys.exit(-1)
'''

#sys.exit(-1)


# model construction

inputs = layers.Input(shape=(140,))
h1     = layers.Dense(800, activation="relu")(inputs)
h2     = layers.Dense(500, activation="relu")(h1)
output = layers.Dense(100, activation="relu")(h1)

encoder = keras.Model(inputs, output, name="aenc_encoder")

inputs = layers.Input(shape=(100,))
h1     = layers.Dense(500, activation="relu")(inputs)
h2     = layers.Dense(800, activation="relu")(h1)
output = layers.Dense(140, activation="sigmoid")(h2)

decoder = keras.Model(inputs, output, name="aenc_decoder")


inputs = layers.Input(shape=(140,))
l1     = encoder(inputs)
output = decoder(l1)

model = keras.Model(inputs, output, name="aenc")

model.summary()

# model settings

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss="mae")


# model training

model.fit(train_x, train_x,
          epochs=20,
          batch_size=64)

# model testing

loss = model.evaluate(test_x, test_x)

print (f"Testing Loss: {loss}")

# model prediction

pred = model.predict(train_x)
train_loss = tf.keras.losses.mae(pred, train_x)

print (f"\nTraining Loss values: {train_loss[:100]}")

threshold = np.mean(train_loss) + np.std(train_loss)
print("\nThreshold: ", threshold)

pred = model.predict(anm_test_x)
test_loss = tf.keras.losses.mae(pred, anm_test_x)

print (f"\nTesting Loss values: {test_loss}")

test_loss = test_loss.numpy()
#test_loss[0] = 0.00152
preds = tf.math.less(test_loss, threshold)
#print (f"preds: {preds}")

print (f"\nAccuracy: {accuracy_score(anm_test_y, preds)}")

