import numpy as np


class Agent:
    def __init__(self, k_arm=10, initial=0.0):
        self.k = k_arm
        self.initial = initial
        self.indices = np.arange(self.k)

    def init(self):
        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # number of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.action = None
        self.time = 0

    def act(self):
        pass

    def step(self, reward):
        action = self.action

        self.time += 1
        self.action_count[action] += 1
        step_size = 1.0 / self.action_count[action]
        self.q_estimation[action] += (reward - self.q_estimation[action]) * step_size


class GreedyAgent(Agent):
    def __init__(self, k_arm=10, initial=0.0):
        super().__init__(k_arm=k_arm, initial=initial)
    
    def act(self):
        q_best = np.max(self.q_estimation)
        self.action = np.random.choice(np.where(self.q_estimation == q_best)[0])

        return self.action


class EpsilonGreedyAgent(GreedyAgent):
    def __init__(self, k_arm=10, initial=0.0, epsilon=0.0):
        super().__init__(k_arm=k_arm, initial=initial)
        self.epsilon = epsilon
    
    def act(self):
        if np.random.rand() < self.epsilon:
            self.action = np.random.choice(self.indices)
            return self.action
        else:
            return super().act()
