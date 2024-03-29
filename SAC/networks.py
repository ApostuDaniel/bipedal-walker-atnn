import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.distributions.normal as Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, name='critic', fc1_dims=256, fc2_dims=256, temp_dir="tmp/sac"):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.temp_dir = temp_dir
        self.chekpoint_file = os.path.join(self.temp_dir, name + "_sac")

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action.reshape(self.fc1_dims, 1)], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q = self.q(action_value)

        return q

    def save_chekpoint(self):
        T.save(self.state_dict(), self.chekpoint_file)

    def load_chekpoint(self):
        self.load_state_dict(T.load(self.chekpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, name='value', fc1_dims=256, fc2_dims=256, temp_dir="tmp/sac"):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.temp_dir = temp_dir
        self.checkpoint_file = os.path.join(self.temp_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_chekpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_chekpoint(self):
        self.load_state_dict(T.load(self.chekpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, name='actor', fc1_dims=256, fc2_dims=256, n_actions=2, temp_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.max_action = max_action
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.temp_dir = temp_dir
        self.chekpoint_file = os.path.join(self.temp_dir, name + "_sac")
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action, dtype=T.float).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_chekpoint(self):
        T.save(self.state_dict(), self.chekpoint_file)

    def load_chekpoint(self):
        self.load_state_dict(T.load(self.chekpoint_file))









