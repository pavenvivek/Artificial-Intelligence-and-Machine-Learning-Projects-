import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

#from sklearn.datasets import fetch_california_housing
from keras import layers

# data preprocessing

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

train_cnt = int(1 * len(x_train))
test_cnt  = int(1 * len(x_test))

x_train = tf.constant(x_train[:train_cnt], dtype=np.float32)
y_train = tf.constant(y_train[:train_cnt])

x_test = tf.constant(x_test[:test_cnt], dtype=np.float32)
y_test = tf.constant(y_test[:test_cnt])

#x_train, x_test = x_train / 255.0, x_test / 255.0
#print (f"----> type : {type(x_train)}")
#x_train, x_test = x_train.astype(float), x_test.astype(float)

#x_train = x_train.reshape(-1, 32, 32, 3)
#x_test = x_test.reshape(-1, 32, 32, 3)

print (f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
#print (f"x_train: {x_train[0].shape}")

# model construction

inputs = layers.Input(shape=(32, 32, 3))
conv1  = layers.Conv2D(32, 3, activation="relu")(inputs)
b1     = layers.BatchNormalization()(conv1)
maxpl  = layers.MaxPooling2D(pool_size=(3, 3))(b1)
conv2  = layers.Conv2D(64, 3, activation="relu")(maxpl)
b2     = layers.BatchNormalization()(conv2)
maxpl  = layers.MaxPooling2D(pool_size=(2, 2))(b2)
conv3  = layers.Conv2D(64, 3, activation="relu")(maxpl)
b3     = layers.BatchNormalization()(conv3)
flt    = layers.Flatten()(b3)
h1     = layers.Dense(500, activation="relu")(flt)
b4     = layers.BatchNormalization()(h1)
outputs = layers.Dense(100, activation="softmax")(b4)


model = keras.Model(inputs=inputs, outputs=outputs, name="cnn")

'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(3, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(10, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(100) # 100 classes in CIFAR-100
])
'''

model.summary()

# model settings

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# model training

model.fit(x_train, y_train,
          epochs=30,
          batch_size=64)
          #validation_data=(x_test, y_test))

# model testing

loss, acc = model.evaluate(x_test, y_test)

print (f"Loss: {loss}, Accuracy: {acc}")

# model prediction

for i in range(0, 10):
    x, y = x_test[i], y_test[i]
    pred = model.predict(np.array([x]))

    print (f"{i} -> pred: {pred.argmax()}, y: {y}")


