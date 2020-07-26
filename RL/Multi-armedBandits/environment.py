import numpy as np


class Environment:
    def __init__(self, k_arm=10):
        self.k = k_arm
    
    def init(self, seed=None):
        self.seed = seed
        np.random.seed(seed)

        self.q_true = np.random.randn(self.k)
    
    def step(self, action):
        reward = self.q_true[action] + np.random.randn()

        return reward