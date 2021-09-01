import random
from collections import deque

import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from skimage.transform import resize
from Config import *

from Model import DeepQModel
from Utils import Utils


class Agent:

    def __init__(self, height, width, env_name='SuperMarioBros-v0'):
        self.env = gym_super_mario_bros.make(env_name)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        self.num_actions = self.env.action_space.n
        # Define state as a queue
        self.state = deque(maxlen=4)
        self.height = height
        self.width = width

        # Initialize state with empty frames
        self.state.append(np.zeros((height, width)))
        self.state.append(np.zeros((height, width)))
        self.state.append(np.zeros((height, width)))
        self.state.append(np.zeros((height, width)))

        self.env.reset()

    def random_action(self):
        return random.randint(0, self.num_actions - 1)

    def play(self, act, curr_time, skip_frame=4):
        current_state = self.state.copy()
        current_state = np.array(current_state)
        current_state = current_state.transpose(1, 2, 0)

        total_reward = 0
        x_pos = 40  # Starting x coordinate
        for _ in range(0, skip_frame):
            state, reward, done, info = self.env.step(act)
            total_reward = total_reward + reward
            # Mario dies or time is up
            if done or info['time'] <= 1 or info['time'] > curr_time:
                # Remove a lot of points if time is still high
                if curr_time > 350:
                    total_reward -= 300
                # Reward for completing the level
                if done:
                    total_reward += 1000
                # Remove some points to not reward the AI immediately
                total_reward -= 70
                # Give more points, the further Mario gets
                total_reward += x_pos / 3
                # The more time is left, the more points are removed
                total_reward -= curr_time / 10
                done = True  # Exit iteration for loop
                break

            # Information from the frame just before the game ends must be gathered before Mario dies
            curr_time = info['time']
            x_pos = info['x_pos']

            # Optional, make training process watchable
            self.env.render()

        state = resize(Utils.pre_process(state), (self.height, self.width), anti_aliasing=True)

        self.state.append(state)
        next_state = self.state.copy()
        next_state = np.array(next_state)
        next_state = next_state.transpose(1, 2, 0)
        return current_state, next_state, total_reward, done, curr_time


def main():
    games_info = []
    # Defining size of the frame by reducing it by half
    img_height = int(224 / 2)
    img_width = int(256 / 2)

    # Define shape of the input - stack of 4 frames
    input_shape = (img_height, img_width, 4)

    # Create agent
    agent = Agent(img_height, img_width)
    output_shape = agent.num_actions
    agent.env.reset()
    agent.env.close()
    #
    model = DeepQModel(input_shape, output_shape, learning_rate=0.1, gamma=0.99)

    # An episode is an individual game
    for episode in range(0, 1000):

        agent = Agent(img_height, img_width)
        current_state = agent.state.copy()
        current_state = np.array(current_state)
        current_state = current_state.transpose(1, 2, 0)
        current_state = np.array([current_state])
        curr_time = 400

        # Further comments and names refer to these four frames as iteration
        for iteration in range(0, 10000):
            game_reward = 0

            if model.epsilon_condition():
                action = agent.random_action()
                Utils.format_action(True, action)
            else:
                action = np.argmax(model.predict([current_state])[0]).item()
                Utils.format_action(False, action)

            current_state, next_state, frame_reward, done, curr_time = agent.play(action, curr_time)
            game_reward += frame_reward
            current_state = np.array([current_state])
            next_state = np.array([next_state])
            model.append_replay((current_state, action, frame_reward, next_state, done))
            current_state = next_state
            if done:
                model.sync_networks()
                break
        # After game ended
        model.train()
        agent.env.reset()
        agent.env.close()
        print("Episode:", episode, "Reward:", game_reward, "Gamma:", model.gamma, "Epsilon:", model.epsilon)

        # Save statistics
        games_info.append(dict.fromkeys(["reward", "epsilon", "gamma"]))
        games_info[episode]['reward'] = game_reward
        games_info[episode]['epsilon'] = model.epsilon
        games_info[episode]['gamma'] = model.gamma


        if episode % 10 == 0 and episode > 0:
            if GRAPH:
                plt.plot([d['reward'] for d in games_info])
                plt.plot([d['epsilon'] for d in games_info])
                plt.show()
            model.save_model()


main()
