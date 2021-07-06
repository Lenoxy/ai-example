import os
from random import randrange
import numpy as np
import tensorflow as tf

MODEL_SAVE_PATH = 'models/tf2_image_ann'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize dataset

x_train = x_train / 255
x_test = x_test / 255

# Reshape all elements (-1) to 28*28 pixels
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

if os.path.exists(MODEL_SAVE_PATH):
    print('Using existing model from', MODEL_SAVE_PATH)
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
else:
    print('No existing model found. Creating a new model...')

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=256, activation='relu', input_shape=(28 * 28,)))

    # Dropout prevents overfitting to train data
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=20, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=8)
    model.save(MODEL_SAVE_PATH)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_accuracy)

sample = randrange(y_test.size)
print("Sample index at", sample)

predictions = model.predict(x_test)
print('Actual:', y_test[sample])
print('AI Prediction:', np.argmax(predictions[sample]))
print('AI Probabilities', np.round(predictions, decimals=3)[sample])
