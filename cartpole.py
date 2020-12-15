import gym
import numpy as np
import math

from agents import QAgent, Agent, RandomAgent, DQNAgent, PrioritizedDQNAgent



env = gym.make('CartPole-v0')

num_episodes = 10000
print_evry= 10
BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 32             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 1e-1                  # for soft update of target parameters
LR = 0.001                  # learning rate
UPDATE_NN_EVERY = 1        # how often to update the network

# prioritized experience replay
UPDATE_MEM_EVERY = 10          # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 30     # how often to update the hyperparameters
EXPERIENCES_PER_SAMPLING = math.ceil(BATCH_SIZE * UPDATE_MEM_EVERY / UPDATE_NN_EVERY)

#agent = Agent(0.1, 1, 1, 0.1, 0.999, env.action_space.n)
#agent = RandomAgent(env.action_space.n)
#agent = QAgent(env, 0.1, 0.9, 1, 0.1, 0.9999, env.action_space.n)
agent = DQNAgent(state_size=env.observation_space.shape[0],
                 action_size=env.action_space.n,
                 buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 gamma=GAMMA,
                 tau=TAU,
                 lr=LR,
                 update_every=UPDATE_NN_EVERY
                )

#agent = PrioritizedDQNAgent(state_size=env.observation_space.shape[0],
#                             action_size=env.action_space.n,
#                             buffer_size=BUFFER_SIZE,
#                             batch_size=BATCH_SIZE,
#                             gamma=GAMMA,
#                             tau=TAU,
#                             lr=LR,
#                             update_every=UPDATE_NN_EVERY,
#                             update_mem_every=UPDATE_MEM_EVERY,
#                             update_mem_par_every=UPDATE_MEM_PAR_EVERY,
#                             experience_per_sampling=EXPERIENCES_PER_SAMPLING
 #                           )

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
        #learning_rate = agent.scheduler.get_lr()[0]
        print(f"Episode {episode}/{num_episodes},eps:{agent.epsilon:.3f}, reward:{reward_last_100}")

        if reward_last_100 >= 195:
            print(f"Solved in {episode} epsiodes")
            break




