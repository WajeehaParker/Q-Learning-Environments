import gym
from IPython.display import clear_output
from time import sleep
import numpy as np
import random

class TaxiEnvironment:
    def __init__(self):
        self.env = gym.make("Taxi-v3", render_mode='ansi').env
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

    def Train(self):
        alpha = 0.1
        gamma = 0.6
        epsilon = 0.1

        for i in range(1, 100001):
            state = self.env.reset()[0]
            epochs, penalties, reward, = 0, 0, 0
            done = False
            
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(self.q_table[state]) # Exploit learned values

                next_state, reward, done, _, info = self.env.step(action) 
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1
                
            if i % 100 == 0:
                clear_output(wait=True)
                print(f"Episode: {i}")

        print("Training finished.\n")

    def Evaluate(self):
        total_epochs, total_penalties = 0, 0
        frames = []  # for animation
        episodes = 100

        for _ in range(episodes):
            state = self.env.reset()[0]
            epochs, penalties, reward = 0, 0, 0
            done = False
            
            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, _, info = self.env.step(action)

                if reward == -10:
                    penalties += 1

                frames.append({
                    'frame': self.env.render(),
                    'state': state,
                    'action': action,
                    'reward': reward
                })

                epochs += 1

            total_penalties += penalties
            total_epochs += epochs

        self.print_frames(frames)
        print(f"Results after {episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / episodes}")
        print(f"Average penalties per episode: {total_penalties / episodes}")

    def print_frames(self, frames):
        for i, frame in enumerate(frames):
            #if(frame['reward'] > -1):
            clear_output(wait=True)
            print(frame['frame'])
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(1)

    def render_environment(self):
        self.env.reset()
        self.env.render()
