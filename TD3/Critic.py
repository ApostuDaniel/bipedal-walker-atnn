import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    l1: nn.Linear
    l2: nn.Linear
    l3: nn.Linear

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_size + action_size, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        Q = F.relu(self.l1(state_action))
        Q = F.relu(self.l2(Q))
        Q = self.l3(Q)
        return Q
