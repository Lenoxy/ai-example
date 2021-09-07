import random
from collections import deque

import math
import numpy as np
import pandas_datareader as data_reader
import tensorflow as tf
from tqdm import tqdm


class AI_Trader:
    def __init__(self, state_size, action_space=3, model_name="AITrader"):  # Stay, Buy, Sell
        self.state_size = state_size
        self.action_space = action_space
        self.model_name = model_name
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.gamma = 0.95
        # To start training, will be adjusted
        self.epsilon = 1
        # Have the trader explore
        self.epsilon_final = 0.01

        self.epsilon_decay = 0.995

        self.model = self.model_builder()

    def model_builder(self):
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))

        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        return model

    # Random or predicted trade
    def trade(self, state):
        if random.random() <= self.epsilon:
            # 0, 1 or 2 for stay, buy or sell
            return random.randrange(self.action_space)
        actions = self.model.predict(state)
        print(actions)
        # Spit out the index with the highest probability (stay, buy or sell)
        return np.argmax(actions)

    def batch_train(self, batch_size):
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)
            target[0][action] = reward

            # Reinforce
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def stocks_price_format(n):
    if n < 0:
        return "- ${0:2f}".format(abs(n))
    else:
        return "${0:2f}".format(abs(n))


def dataset_loader(stock_name):
    # Complete the dataset loader function
    dataset = data_reader.DataReader(stock_name, data_source="yahoo")

    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[-1]).split()[0]

    close = dataset['Close']

    return close


def create_state(data, timestep, window_size):
    starting_id = timestep - window_size + 1

    if starting_id >= 0:
        windowed_data = data[starting_id:timestep + 1]
    else:
        windowed_data = - starting_id * [data[0]] + list(data[0:timestep + 1])

    state = []
    # normalize prices to be the differences instead of actual values
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))

    return np.array([state])


stock_name = "AAPL"
data = dataset_loader(stock_name)
print(data)

window_size = 10
episodes = 1000
batch_size = 32
data_samples = len(data) - 1

trader = AI_Trader(window_size)
trader.model.summary()

for episode in range(episodes + 1):
    print("Episode {}/{}".format(episode, episodes))
    state = create_state(data, 0, window_size + 1)
    total_profit = 0
    trader.inventory = []
    print("\n")
    for time in tqdm(range(data_samples)):
        action = trader.trade(state)

        next_state = create_state(data, time + 1, window_size)
        reward = 0

        if action == 1:  # Buy
            trader.inventory.append(data[time])
            print("AI Trader bought for: ", stocks_price_format(data[time]))


        elif action == 2 and len(trader.inventory) > 0:  # Sell
            buy_price = trader.inventory.pop(0)
            # max(amount of money made or zero)
            reward = max(data[time] - buy_price, 0)
            total_profit += data[time] - buy_price
            print("AI Trader sold for: ", stocks_price_format(data[time]),
                  " Profit: " + stocks_price_format(data[time] - buy_price))

    done = time == data_samples - 1

    trader.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
        print("########################")
        print("TOTAL PROFIT IN EPISODE {}: {}".format(episode, total_profit))
        print("########################")

    if len(trader.memory) > batch_size:
        trader.batch_train(batch_size)

    if episode % 10 == 0:
        trader.model.save("models/ai_trader.h5")
