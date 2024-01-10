import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    l1: nn.Linear
    l2: nn.Linear
    l3: nn.Linear
    max_action: float

    def __init__(self, state_size, action_size, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_size, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_size)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x
