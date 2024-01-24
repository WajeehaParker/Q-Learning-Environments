import numpy as np
import gym

class MountainCar:
    def __init__(self, learning, discount, epsilon, min_eps, episodes):
        #self.env = gym.make('MountainCar-v0', render_mode="human")
        self.env = gym.make('MountainCar-v0')
        self.learning = learning
        self.discount = discount
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.episodes = episodes

        self.num_states = (self.env.observation_space.high - self.env.observation_space.low) * np.array([10, 100])
        self.num_states = np.round(self.num_states, 0).astype(int) + 1

        self.Q = np.random.uniform(low=-1, high=1, size=(self.num_states[0], self.num_states[1], self.env.action_space.n))

    def discretize_state(self, state):
        state_adj = (state - self.env.observation_space.low) * np.array([10, 100])
        return np.round(state_adj, 0).astype(int)

    def train(self):
        reward_list = []
        ave_reward_list = []
        reduction = (self.epsilon - self.min_eps) / self.episodes

        for i in range(self.episodes):
            done = False
            tot_reward, reward = 0, 0
            state = self.env.reset()[0]
            state_adj = self.discretize_state(state)

            while not done:
                if i >= (self.episodes - 20):
                    self.env.render()

                if np.random.random() < 1 - self.epsilon:
                    action = np.argmax(self.Q[state_adj[0], state_adj[1]])
                else:
                    action = np.random.randint(0, self.env.action_space.n)

                state2, reward, done, _, _ = self.env.step(action)
                state2_adj = self.discretize_state(state2)

                if done and state2[0] >= 0.5:
                    self.Q[state_adj[0], state_adj[1], action] = reward
                else:
                    delta = self.learning * (reward + self.discount * np.max(self.Q[state2_adj[0], state2_adj[1]]) -
                                             self.Q[state_adj[0], state_adj[1], action])
                    self.Q[state_adj[0], state_adj[1], action] += delta

                tot_reward += reward
                state_adj = state2_adj

            if self.epsilon > self.min_eps:
                self.epsilon -= reduction

            reward_list.append(tot_reward)

            if (i + 1) % 100 == 0:
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                reward_list = []

                print('Episode {} Average Reward: {}'.format(i + 1, ave_reward))

        self.env.close()
        return ave_reward_list
    
    def test(self, num_episodes=10):
        total_rewards = []

        for _ in range(num_episodes):
            state = self.env.reset()[0]
            state_adj = self.discretize_state(state)
            done = False
            episode_reward = 0

            while not done:
                self.env.render()  # Comment this line if you don't want to visualize the test

                action = np.argmax(self.Q[state_adj[0], state_adj[1]])
                state, reward, done, _, _ = self.env.step(action)
                state_adj = self.discretize_state(state)
                episode_reward += reward

            total_rewards.append(episode_reward)

        self.env.close()
        avg_reward = np.mean(total_rewards)

        print(f'Testing finished.\nAverage Reward over {num_episodes} episodes: {avg_reward}')