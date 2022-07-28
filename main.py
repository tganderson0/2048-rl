import gym_2048
import gym
from modules.Agent import Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
  env = gym.make('2048-v0')

  agent = Agent()

  avg_reward = agent.train(env, load_weights=True)

  plt.plot(avg_reward)
  plt.xlabel("Episode")
  plt.ylabel("Avg. Episodic Reward")
  plt.show()

