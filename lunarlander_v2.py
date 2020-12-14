import gym
import numpy as np

from agents import QAgent, Agent, RandomAgent, DQNAgent


env = gym.make('LunarLander-v2')

num_episodes = 5000

agent = DQNAgent(env.observation_space.n, env.action_space.n)


average_reward = []
for episode in range(num_episodes):
    rewards = []
    state = env.reset()


    while True:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        agent.step(state, action, reward, next_state, done)
        state = next_state

        if done:
            average_reward.append(np.sum(rewards))
            break

#

    # monitor progress
    if episode % 500 == 0:
        print(f"Episode {episode}/{num_episodes}, reward:{int(np.mean(average_reward[-99:]))}")
        #sys.st_out.flush()
