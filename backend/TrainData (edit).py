import tensorflow as tf
import pandas as pd
import numpy as np

from LoadData import x, y

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, SGD
from keras.losses import CategoricalCrossentropy

tf.keras.backend.clear_session()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(y_train.shape, x_train.shape)

tf.keras.backend.clear_session()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(y_train.shape, x_train.shape)

model = Sequential([
    Conv2D(2, (2, 2), strides=(2, 2), padding='same', activation='relu', input_shape=(586, 1048, 3)),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(2, activation='softmax'),
    Dense(5, activation='relu', use_bias=True)
])

model.summary()
model.compile(optimizer=SGD(learning_rate=0.01), loss=CategoricalCrossentropy(), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=1)
model.save('model_v3(obj_d)', save_format='h5')

model = Sequential([
    Conv2D(2, (2, 2), strides=(2, 2), padding='same', activation='relu', input_shape=(586, 1048, 3)),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(2, activation='softmax'),
    Dense(5, activation='relu', use_bias=True)
])

model.summary()
model.compile(optimizer=SGD(learning_rate=0.01), loss=CategoricalCrossentropy(), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=1)
model.save('model_v3(obj_d)', save_format='h5')
