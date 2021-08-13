import os
import random
from collections import deque

import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import tensorflow as tf

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

MODEL_SAVE_PATH = "models/tf2_mario_bros_r"


class Player:
    def __init__(self, state_size, action_space=7, model_name="Player"):
        self.state_size = state_size
        self.action_space = action_space
        self.model_name = model_name
        self.gamma = 0.95
        self.memory = deque(maxlen=100)
        # To start training, will be adjusted
        self.epsilon = 1
        # Have the player explore
        self.epsilon_final = 0.05
        self.epsilon_decay = 0.995

        self.model = self.model_builder()

    def model_builder(self):
        if os.path.exists(MODEL_SAVE_PATH):
            print('Using existing model from', MODEL_SAVE_PATH)
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        else:
            print('No existing model found. Creating a new model...')
            model = tf.keras.models.Sequential()

            model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
            model.add(tf.keras.layers.Dense(units=64, activation='relu'))
            model.add(tf.keras.layers.Dense(units=128, activation='relu'))

            model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))

            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))

        return model

    # Random or predicted trade
    def play(self, state):
        if random.random() <= self.epsilon:
            # 0, 1 or 2 for stay, buy or sell
            return random.randrange(self.action_space)
        actions = self.model.predict(state)
        print(actions)
        # Spit out the index with the highest probability (stay, buy or sell)
        return np.argmax(actions)

    def train(self, max_episodes=1000, batch_size=1000):
        for ep in range(max_episodes):
            done = False
            total_reward = 0
            state = env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, info = env.step(action)
                self.memory.append((state, action, reward*0.01, next_state, done))
                total_reward += reward
                state = next_state
            print('EP{} EpisodeReward={}'.format(ep, total_reward))


player = Player(50)
for rounds in range(10):  # Retries
    done = True
    for step in range(5000):  # 5000 Frames per Game
        if done:
            state = env.reset()

        action = player.play(state)
        print(action)
        state, reward, done, info = env.step(action)
        print("reward ", reward)
        env.render()
    player.model.save(MODEL_SAVE_PATH)
