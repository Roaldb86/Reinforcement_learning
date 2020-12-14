from collections import defaultdict, namedtuple, deque
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models import QNetwork
from utilities import hash_state, discretize, create_uniform_grid

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

    def act(self, state):
        state = hash_state(state)
        probs = self.get_probs(state)
        action = np.random.choice(np.arange(self.nA), p=probs)
        return action

    def step(self, state, action, reward, next_state, done):
        state = hash_state(state)
        next_state = hash_state((next_state))

        probs = self.get_probs(state)

        if not done:
            self.Q[state][action] += self.alpha * (
                    reward + self.gamma * (np.dot(self.Q[next_state], probs)) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])



class ExpectedSarsaAgent:

    def __init__(self,
                 nA=6,
                 alpha=0.20,
                 epsilon_max=1,
                 epsilon_min=0.00005,
                 eps_decay=0.999,
                 gamma=0.95
                 ):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay
        self.epsilon = epsilon_max
        self.gamma = gamma

    def get_probs(self, state,  eps=None):
        self.epsilon = max(self.epsilon * self.eps_decay, self.epsilon_min)
        if eps:
            self.epsilon = eps
        best_a = np.argmax(self.Q[state])
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[best_a] += 1 - self.epsilon
        return policy_s

    def act(self, state, episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = self.get_probs(state)

        if state in self.Q:
            return np.random.choice(np.arange(self.nA), p=probs)
        else:
            return np.random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done, episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        probs = self.get_probs(state)

        if not done:
            self.Q[state][action] += self.alpha * (
                        reward + self.gamma * (np.dot(self.Q[next_state], probs)) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])


class QAgent:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, eps_decay, nA):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.high = env.env.observation_space.high
        self.low = env.observation_space.low
        self.grid = create_uniform_grid(self.low, self.high)

    def get_probs(self, state):

        self.epsilon = max(self.epsilon * self.eps_decay, self.epsilon_min)
        policy = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        policy[best_a] += 1 - self.epsilon
        return policy

    def act(self, state):
        state = discretize(state, self.grid)
        state = hash_state(state)
        probs = self.get_probs(state)
        action = np.random.choice(np.arange(self.nA), p=probs)
        return action

    def step(self, state, action, reward, next_state, done):
        state = discretize(next_state, self.grid)
        state = hash_state(state)
        next_state = hash_state((next_state))

        if not done:
            self.Q[state][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])


class RandomAgent:
    def __init__(self, nA):
        self.nA = nA

    def act(self, _):
        return np.random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        pass


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.9  # discount factor
TAU = 1e-3 # for soft update of target parameters
LR = 0.01  # learning rate
UPDATE_EVERY = 1  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 seed=42,
                 epsilon=1,
                 epsilon_min=0.05,
                 eps_decay=0.9999
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.learn_steps = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.999)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, state, action, reward, next_state,  done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        self.epsilon = max(self.epsilon*self.eps_decay, self.epsilon_min)
        self.scheduler.step()

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            #print(action_values)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            #print(np.argmax(action_values.cpu().data.numpy()))
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)