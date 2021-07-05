import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

plt.imshow(x_train[10])
plt.show()

# Normalize dataset

x_train = x_train / 255
x_test = x_test / 255

# Reshape all elements (-1) to 28*28 pixels
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=256, activation='relu', input_shape=(28 * 28,)))

# Dropout prevents overfitting to train data
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=20, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, epochs=10)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test accuracy: ' + str(test_accuracy))

prediction = model.evaluate(x_test)

print(prediction)
