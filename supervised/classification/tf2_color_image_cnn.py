import os
from random import randrange

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

MODEL_SAVE_PATH = 'models/tf2_color_image_cnn'

class_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# All values are now between 0 and 1 (due to this being 8-bit color images), which apparently helps the neural network
# (Data normalization)
x_train = x_train / 255.0
x_test = x_test / 255.0

# print(x_train.shape)
# (50000, 32, 32, 3)
# (Amount of images, pixels horizontal, pixels vertical, color channels)

if os.path.exists(MODEL_SAVE_PATH):
    print('Using existing model from', MODEL_SAVE_PATH)
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
else:
    print('No existing model found. Creating a new model...')
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
    # Flatten to receive a 2D classification (just like the TF playground)
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=128, activation="relu"))
    model.add(tf.keras.layers.Dense(units=10, activation="softmax"))
    model.summary()

    adam = tf.keras.optimizers.Adam()
    model.compile(loss="sparse_categorical_crossentropy", optimizer=adam, metrics=["sparse_categorical_crossentropy"])
    model.fit(x_train, y_train, epochs=5, batch_size=8)
    model.save(MODEL_SAVE_PATH)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test accuracy:", str(test_accuracy))

sample = randrange(y_test.size)
print("Sample index at", sample)

predictions = model.predict(x_test)
print('Actual:', class_names[y_test[sample]][0])
print('AI Prediction:', class_names[np.argmax(predictions[sample])])
formatted_probabilities = np.vstack((class_names, np.round(predictions, decimals=3)[sample])).T
print("AI Probabilities:\n", formatted_probabilities)
plt.imshow(x_test[sample])
plt.show()
