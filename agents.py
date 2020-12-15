from collections import defaultdict, namedtuple, deque
import numpy as np
import random
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models import QNetwork
from utilities import hash_state, discretize, create_uniform_grid
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedDQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 lr,
                 update_every,
                 update_mem_every,
                 update_mem_par_every,
                 experience_per_sampling,
                 seed=25,
                 epsilon=1,
                 epsilon_min=0.01,
                 eps_decay=0.999,
                 compute_weights=False
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
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.experience_per_sampling  = experience_per_sampling
        self.update_mem_every = update_mem_every
        self.update_mem_par_every = update_mem_par_every
        self.seed = random.seed(seed)
        self.learn_steps = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay
        self.compute_weights = compute_weights

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.995)


        # Replay memory
        self.memory = PrioritizedReplayBuffer(
                    self.action_size,
                    self.buffer_size,
                    self.batch_size,
                    self.experience_per_sampling,
                    self.seed,
                    self.compute_weights)
        # Initialize time step (for updating every UPDATE_NN_EVERY steps)
        self.t_step_nn = 0
        # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
        self.t_step_mem_par = 0
        # Initialize time step (for updating every UPDATE_MEM_EVERY steps)
        self.t_step_mem = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_NN_EVERY time steps.
        self.t_step_nn = (self.t_step_nn + 1) % self.update_every
        self.t_step_mem = (self.t_step_mem + 1) % self.update_mem_every
        self.t_step_mem_par = (self.t_step_mem_par + 1) % self.update_mem_par_every
        if self.t_step_mem_par == 0:
            self.memory.update_parameters()
        if self.t_step_nn == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.experience_count > self.experience_per_sampling:
                sampling = self.memory.sample()
                self.learn(sampling)
        if self.t_step_mem == 0:
            self.memory.update_memory_sampling()

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        self.epsilon = max(self.epsilon*self.eps_decay, self.epsilon_min)

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

    def learn(self, sampling):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            sampling (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices  = sampling

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        if self.compute_weights:
            with torch.no_grad():
                weight = sum(np.multiply(weights, loss.data.cpu().numpy()))
            loss *= weight

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.learn_steps += 1

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        # ------------------- update priorities ------------------- #
        delta = abs(Q_targets - Q_expected.detach()).numpy()
        self.memory.update_priorities(delta, indices)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 lr,
                 update_every,
                 seed=22,
                 epsilon=1,
                 epsilon_min=0.05,
                 eps_decay=0.99
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
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.seed = random.seed(seed)
        self.learn_steps = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, state, action, reward, next_state,  done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # sample
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
        """
        self.epsilon = max(self.epsilon*self.eps_decay, self.epsilon_min)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
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
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

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
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
