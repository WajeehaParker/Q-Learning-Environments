import subprocess
from MountainCar import MountainCar
import matplotlib.pyplot as plt
import numpy as np

try:
    subprocess.run(['python', 'MountainCar.py'], check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

# Instantiate and run the Q-learning agent
agent = MountainCar(learning=0.2, discount=0.9, epsilon=0.8, min_eps=0, episodes=5000)
rewards = agent.train()

# Plot Rewards
plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.jpg')
plt.close()

agent.test()
