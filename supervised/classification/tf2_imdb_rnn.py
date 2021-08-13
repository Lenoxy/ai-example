# TODO fix

import sys
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=sys.maxsize)

number_of_words = 20000
vocab_size = number_of_words


max_len = 100
embed_size = 128


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=number_of_words)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, embed_size, input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=128)
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test accuracy: {}".format(test_accuracy))