
import gym
import numpy as np

from agents import QAgent, Agent, RandomAgent, DQNAgent


env = gym.make('MountainCar-v0')

num_episodes = 500
print_evry = 100
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)


average_reward = []
for episode in range(1, num_episodes):
    rewards = []
    state = env.reset()

    # max steps
    while True:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        agent.step(state, action, reward, next_state, done)
        state = next_state

        if done:
            average_reward.append(np.sum(rewards))
            break


    # monitor progress
    if episode % print_evry == 0:
        print(f"Episode {episode}/{num_episodes}, reward:{int(np.mean(average_reward[-99:]))}")
        #sys.st_out.flush()
