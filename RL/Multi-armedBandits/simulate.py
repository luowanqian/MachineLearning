import numpy as np


def simulate(runs, times, bandits, envs):
    rewards = np.zeros((len(bandits), runs, times))
    for i, bandit in enumerate(bandits):
        env = envs[i]
        for r in range(runs):
            bandit.init()
            env.init()
            for t in range(times):
                action = bandit.act()
                reward = env.step(action)
                bandit.step(reward)
                rewards[i, r, t] = reward
    mean_rewards = rewards.mean(axis=1)
    return mean_rewards