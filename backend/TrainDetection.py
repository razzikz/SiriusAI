from LoadData import x_tensor, y_tensor, label

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Sequential
import matplotlib.pyplot as plt
from keras.src.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.src.optimizers import Adam

tf.keras.backend.clear_session()

x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=8)

model = Sequential([
    Conv2D(64, (2, 2), strides=(2, 2), padding='same', activation='relu', input_shape=(825, 550, 3)),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (1, 1), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (1, 1), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(1470, activation='linear')
])

model.summary()
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=16)
model.save('model_v1', save_format='h5')
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print(f"Accuracy: {accuracy}\nLoss: {loss}")

plt.plot(accuracy, 'g', label='Accuracy')
plt.plot(loss, 'r', label='Loss')
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()