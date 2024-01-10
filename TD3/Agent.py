import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from Actor import Actor
from Critic import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    actor: Actor
    actor_target: Actor
    actor_optimizer: optim.Adam
    critic_1: Critic
    critic_1_target: Critic
    critic_1_optimizer: optim.Adam
    critic_2: Critic
    critic_2_target: Critic
    critic_2_optimizer: optim.Adam
    max_action: float

    def __init__(self, state_size, action_size, max_action, lr=0.001):

        self.actor = Actor(state_size, action_size, max_action).to(device)
        self.actor_target = Actor(state_size, action_size, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_size, action_size).to(device)
        self.critic_1_target = Critic(state_size, action_size).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)

        self.critic_2 = Critic(state_size, action_size).to(device)
        self.critic_2_target = Critic(state_size, action_size).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, memory, current_timestep, batch_size=100,
               gamma=0.99, tau=0.995, noise=0.2, noise_clip=0.5, policy_delay=2):
        for i in range(current_timestep):
            state, action, reward, next_state, done = memory.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)

            noise = torch.FloatTensor(action).data.normal_(0, noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * gamma * target_Q).detach()

            Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            if i % policy_delay == 0:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_((tau * target_param.data) + ((1 - tau) * param.data))

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_((tau * target_param.data) + ((1 - tau) * param.data))

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_((tau * target_param.data) + ((1 - tau) * param.data))

    def save(self, dir, file):
        torch.save(self.actor.state_dict(), dir + '/' + file + '_actor.pth')
        torch.save(self.actor_target.state_dict(), dir + '/' + file + '_actor_target.pth')

        torch.save(self.critic_1.state_dict(), dir + '/' + file + '_critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), dir + '/' + file + '_critic_1_target.pth')

        torch.save(self.critic_2.state_dict(), dir + '/' + file + '_critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), dir + '/' + file + '_critic_2_target.pth')

    def load(self, dir, file):
        self.actor.load_state_dict(
            torch.load(dir + '/' + file + '_actor.pth', map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load(dir + '/' + file + '_actor_target.pth', map_location=lambda storage, loc: storage))

        self.critic_1.load_state_dict(
            torch.load(dir + '/' + file + '_critic_1.pth', map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(
            torch.load(dir + '/' + file + '_critic_1_target.pth', map_location=lambda storage, loc: storage))

        self.critic_2.load_state_dict(
            torch.load(dir + '/' + file + '_critic_2.pth', map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(
            torch.load(dir + '/' + file + '_critic_2_target.pth', map_location=lambda storage, loc: storage))

