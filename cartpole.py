import gym
import numpy as np

from agents import QAgent, Agent, RandomAgent, DQNAgent


env = gym.make('CartPole-v0')

num_episodes = 10000
print_evry= 100

#agent = Agent(0.1, 1, 1, 0.1, 0.999, env.action_space.n)
#agent = RandomAgent(env.action_space.n)
#agent = QAgent(env, 0.1, 0.9, 1, 0.1, 0.9999, env.action_space.n)
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

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

    # monitor progress
    if episode % print_evry == 0:
        reward_last_100 = int(np.mean(average_reward[-99:]))
        learning_rate = agent.scheduler.get_lr().squeeze()
        print(f"Episode {episode}/{num_episodes},eps:{agent.epsilon:.3f}, lr: {learning_rate}, reward:{reward_last_100}")

        if reward_last_100 >= 195:
            print(f"Solved in {episode} epsiodes")
            break




