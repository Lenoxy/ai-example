import random
from collections import deque

import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from skimage.transform import resize

from Model import DeepQModel
from Utils import Utils

# Comment out to disable saving and loading
SAVE_PATH = "./models/mario_bros"


# SAVE_PATH = False


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
        # Initialize values
        x_pos = 40  # Starting point
        curr_time = 400  # Starting time
        for _ in range(0, skip_frame):
            state, reward, done, info = self.env.step(act)
            total_reward = total_reward + reward


            # Mario dies or time is up
            if done or info['time'] <= 1 or info['time'] > curr_time:
                # Remove a lot of points if time is still high
                if curr_time > 350:
                    total_reward -= 300
                # Remove some points to not reward the AI immediately
                total_reward -= 70
                # Give more points, the further Mario gets
                total_reward += x_pos / 4
                # The more time is left, the more points are removed
                total_reward -= curr_time / 10
                done = True
                break

            # Information from the frame just before the game ends must be gathered before Mario dies
            curr_time = info['time']
            x_pos = info['x_pos']

        state = resize(Utils.pre_process(state), (self.height, self.width), anti_aliasing=True)

        self.state.append(state)
        next_state = self.state.copy()
        next_state = np.array(next_state)
        next_state = next_state.transpose(1, 2, 0)
        return current_state, next_state, total_reward, done, curr_time


def main():
    game_rewards = []
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
    model = DeepQModel(input_shape, output_shape, learning_rate=0.1, gamma=0.995, save_path=SAVE_PATH)

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
            action = agent.random_action() if model.epsilon_condition() else \
                np.argmax(model.predict([current_state])[0])

            current_state, next_state, frame_reward, done, curr_time = agent.play(action, curr_time)
            game_reward += frame_reward
            # Optional, make training process watchable
            agent.env.render()
            current_state = np.array([current_state])
            next_state = np.array([next_state])
            model.append_replay((current_state, action, frame_reward, next_state, done))
            current_state = next_state
            if done:
                print("Episode", episode, game_reward, model.gamma)
                game_rewards.append(game_reward)
                print("Rewards:", game_rewards)
                model.sync_networks()
                break
        # After game ended
        model.train()
        agent.env.reset()
        agent.env.close()

        if episode % 10 == 0:
            model.save_model()


main()
