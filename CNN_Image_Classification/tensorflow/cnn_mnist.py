import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

#from sklearn.datasets import fetch_california_housing
from keras import layers

# data preprocessing

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_cnt = int(.10 * len(x_train))
test_cnt  = int(.10 * len(x_test))

x_train = x_train[:train_cnt]
y_train = y_train[:train_cnt]

x_test = x_test[:test_cnt]
y_test = y_test[:test_cnt]

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print (f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")


# model construction

inputs = layers.Input(shape=(28, 28, 1))
conv1  = layers.Conv2D(1, 3, activation="relu")(inputs)
conv2  = layers.Conv2D(1, 3, activation="relu")(conv1)
maxpl  = layers.MaxPooling2D(2,2)(conv2)
#globpl = layers.GlobalAveragePooling2D()(maxpl)
flt    = layers.Flatten()(maxpl)
mlp    = layers.Dense(800, activation="relu")(flt)
outputs = layers.Dense(10, activation="softmax")(mlp)

model = keras.Model(inputs=inputs, outputs=outputs, name="cnn")

model.summary()

# model settings

model.compile(optimizer="Adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])

# model training

model.fit(x_train, y_train,
          epochs=10,
          batch_size=100)

# model testing

loss, acc = model.evaluate(x_test, y_test)

print (f"Loss: {loss}, Accuracy: {acc}")

# model prediction

for i in range(0, 10):
    pred = model.predict(np.array([x_test[i]]))
    print (f"pred: {pred.argmax()}, y: {y_test[i]}")
