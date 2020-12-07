import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6, alpha=0.20, epsilon_max=1, epsilon_min=0.00005, eps_decay=0.999):
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

    def get_probs(self, state, eps=None):
        self.epsilon = max(self.epsilon * self.eps_decay, self.epsilon_min)
        if eps:
            self.epsilon = eps
        best_a = np.argmax(self.Q[state])
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[best_a] += 1 - self.epsilon
        return policy_s

    def select_action(self, state, episode):
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
            self.Q[state][action] += self.alpha * (reward + np.dot(self.Q[next_state], probs) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
