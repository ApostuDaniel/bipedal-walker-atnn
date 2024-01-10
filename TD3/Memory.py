import numpy as np


class Memory:
    memory: list
    max_size: int
    size: int

    def __init__(self, max_size=5e5):
        self.memory = []
        self.max_size = int(max_size)
        self.size = 0

    def save_experience(self, experience):
        self.size += 1
        self.memory.append(experience)

    def sample(self, batch_size):
        if self.size > self.max_size:
            trim_size = min(int(self.size / 5), len(self.memory))
            self.memory = self.memory[trim_size:]
            self.size = len(self.memory)

        sampled_indices = np.random.randint(0, len(self.memory), size=batch_size)
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = [], [], [], [], []

        for i in sampled_indices:
            s, a, r, n_s, d = self.memory[i]
            sampled_states.append(np.array(s, copy=False))
            sampled_actions.append(np.array(a, copy=False))
            sampled_rewards.append(np.array(r, copy=False))
            sampled_next_states.append(np.array(n_s, copy=False))
            sampled_dones.append(np.array(d, copy=False))

        return np.array(sampled_states), np.array(sampled_actions), np.array(sampled_rewards), \
               np.array(sampled_next_states), np.array(sampled_dones)