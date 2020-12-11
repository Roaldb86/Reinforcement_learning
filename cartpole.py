import gym
import numpy as np
from collections import defaultdict

env = gym.make('CartPole-v0')
high = env.observation_space.high
low = env.observation_space.low


def create_uniform_grid(low, high, bins=(10, 10, 10, 10)):
    """
    Define a uniformly-spaced grid that can be used to discretize a space.

    """
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]

    return grid


def discretize(sample, grid):
    """Discretize a sample as per given grid.
    """
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension


class Agent:
    def __init__(self, alpha, gamma, epsilon, epsilon_min, eps_decay, nA):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def get_probs(self, state):

        self.epsilon = max(self.epsilon * self.eps_decay, self.epsilon_min)
        policy = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        policy[best_a] += 1 - self.epsilon
        return policy

    def select_action(self, state):
        state = hash_state(state)
        probs = self.get_probs(state)
        action = np.random.choice(np.arange(self.nA), p=probs)
        return action

    def step(self, state, action, next_state, reward, done):
        state = hash_state(state)
        next_state = hash_state((next_state))

        probs = self.get_probs(state)

        if not done:
            self.Q[state][action] += self.alpha * (
                    reward + self.gamma * (np.dot(self.Q[next_state], probs)) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])

class QAgent(Agent):
    def __init__(self, alpha, gamma, epsilon, epsilon_min, eps_decay, nA):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def get_probs(self, state):

        self.epsilon = max(self.epsilon * self.eps_decay, self.epsilon_min)
        policy = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        policy[best_a] += 1 - self.epsilon
        return policy

    def select_action(self, state):
        state = hash_state(state)
        probs = self.get_probs(state)
        action = np.random.choice(np.arange(self.nA), p=probs)
        return action

    def step(self, state, action, next_state, reward, done):
        state = hash_state(state)
        next_state = hash_state((next_state))

        probs = self.get_probs(state)

        if not done:
            self.Q[state][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])



class RandomAgent:
    def __init__(self, nA):
        self.nA = nA

    def select_action(self, _):
        return np.random.choice(np.arange(self.nA))

    def step(self, state, action, next_state, reward, done):
        pass

def hash_state(state):
    return str(state)

num_episodes = 20000
grid = create_uniform_grid(low, high)
#agent = Agent(0.1, 1, 1, 0.1, 0.999, env.action_space.n)
#agent = RandomAgent(env.action_space.n)
agent = QAgent(0.1, 0.9, 1, 0.1, 0.9999, env.action_space.n)

average_reward = []
for episode in range(num_episodes):
    rewards = []
    state = env.reset()
    state = discretize(state, grid)

    while True:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = discretize(next_state, grid)
        rewards.append(reward)
        agent.step(state, action, next_state, reward, done)
        state = next_state

        if done:
            average_reward.append(np.sum(rewards))
            break

#

    # monitor progress
    if episode % 500 == 0:
        print(f"Episode {episode}/{num_episodes}, reward:{int(np.mean(average_reward[-99:]))}")
        #sys.st_out.flush()



