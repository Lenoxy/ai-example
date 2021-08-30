import os
from collections import deque
from random import sample

import numpy as np
import tensorflow as tf


class DeepQModel:
    """
        Description
        -----------
            the class deals with building the right parameters and functions to
            built a deep learning agent that learn and play games
        Parameters
        ----------
        input_shape : Integer
            the shape for the input states
        output_shape : Integer
            the number of possible actions
        learning_rate : Double
            learning rate for the optimizer for neural network
        gamma : Double
            discount factor to be multiplied with the future rewards
    """

    def __init__(self, input_shape, output_shape, learning_rate: float, gamma: float, save_path = False):
        """
        Description
        -----------
        Initialize shapes, learning_rate, decay and epsilon
        """
        self.save_path = save_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.buffer = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = 1.0
        if not save_path:
            print("Saving Disabled, creating new temporary Model")
            self.predictionModel = self.build_model()
        elif os.path.exists(self.save_path):
            print("Loading model from", self.save_path)
            self.predictionModel = self.load_model(self.save_path)
        else:
            print("No model found at", self.save_path)
            self.predictionModel = self.build_model()
        self.targetModel = self.build_model()
        self.targetModel.set_weights(self.predictionModel.get_weights())

    def sync_networks(self):
        self.targetModel.set_weights(self.predictionModel.get_weights())

    def predict(self, state):
        return self.predictionModel.predict(state)

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, 8, strides=(4, 4), padding='valid', input_shape=self.input_shape,
                                         activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, 4, strides=(2, 2), padding='valid', activation='relu'))
        model.add(tf.keras.layers.Conv2D(128, 3, strides=(1, 1), padding='valid', activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.output_shape, activation='linear'))
        model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def append_replay(self, data):
        self.buffer.append(data)

    def epsilon_decay(self):
        if self.epsilon > 0.01:
            self.epsilon = 0.97 * self.epsilon
        else:
            self.epsilon = 0.01
        print('Epsilon decayed to', self.epsilon)

    def epsilon_condition(self):
        return np.random.rand() <= self.epsilon

    def load_model(self, path):
        model = tf.keras.models.load_model(path)
        return model

    def save_model(self):
        if self.save_path:
            print("Saving...")
            self.predictionModel.save(self.save_path)
            print("Done.")


    def train(self, batch_size=64):
            if batch_size < len(self.buffer):
                samples = sample(self.buffer, batch_size)
            else:
                samples = self.buffer
            for observation in samples:
                state, action, reward, next_state, done = observation
                if done is True:
                    t = reward
                else:
                    next_state_max_reward = np.amax(self.targetModel.predict(next_state))
                    t = reward + (self.gamma * next_state_max_reward)

                target_action_pair = self.predictionModel.predict(state)
                target_action_pair[0][action] = t * (-1)
                self.predictionModel.fit(state, target_action_pair, epochs=1, verbose=0)
            self.epsilon_decay()
