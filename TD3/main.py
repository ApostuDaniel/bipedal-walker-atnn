import torch
import gym
import numpy as np
from Agent import Agent
from Memory import Memory

ENV_NAME = "BipedalWalker-v3"
RENDER = False
RENDER_MODE = 'human'
LOG_AT = 1

RANDOM_SEED = 0
GAMMA = 0.99  # discount for future rewards - used in the Bellman equation
BATCH_SIZE = 100  # num of transitions sampled from replay buffer
LR = 0.001

TAU = 0.995  # target policy update parameter (1-tau)
EXPLORATION_NOISE = 0.1  # Std of Gaussian exploration noise
NOISE = 0.2  # target policy smoothing noise
NOISE_CLIP = 0.5
POLICY_DELAY = 2  # delayed policy updates parameter

MAX_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 2000

DIR = "./saved"
FILE = "TD3_" + ENV_NAME + "_" + str(RANDOM_SEED)

START_EPISODE = 1


def train():
    if not RENDER:
        env = gym.make(ENV_NAME)
    else:
        env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    action_low = env.action_space.low
    action_high = env.action_space.high

    memory = Memory()
    agent = Agent(state_size, action_size, max_action, LR)

    # comment this line if you don't want to load a previous model, also change START_EPISODE to 1
    #agent.load(DIR, FILE + '_solved')

    average_reward = 0
    episode_reward = 0
    fd = open("log.txt", "w+")

    # training procedure:
    for episode in range(START_EPISODE, MAX_EPISODES + 1):
        state = env.reset()
        state = state[0]

        for timestep in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            action = action + np.random.normal(0, EXPLORATION_NOISE, size=env.action_space.shape[0])
            action = action.clip(action_low, action_high)

            next_state, reward, done, truncated, _ = env.step(action)
            memory.save_experience((state, action, reward, next_state, float(done)))
            state = next_state

            average_reward += reward
            episode_reward += reward


            # if episode is done then update agent:
            if done or truncated or timestep == (MAX_STEPS_PER_EPISODE - 1):
                agent.update(memory, timestep, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU,
                             start_noise=NOISE, noise_clip=NOISE_CLIP, policy_delay=POLICY_DELAY)
                break

        fd.write(str(episode) + ' ' + str(episode_reward) + '\n')
        fd.flush()
        episode_reward = 0

        # if average reward for the LOG_AT episodes > 300 then save and stop traning:
        if (average_reward / LOG_AT) >= 300:
            print("Score over 300. Saving model.")
            name = FILE + '_solved'
            agent.save(DIR, name)
            fd.close()
            break

        if episode > 300:
            agent.save(DIR, FILE)
            if episode % 50 == 0:
                agent.save(DIR, FILE + '_' + str(episode))

        if episode % LOG_AT == 0:
            average_reward = int(average_reward / LOG_AT)
            print("Episode: {}\tAverage Reward: {}".format(episode, average_reward))
            average_reward = 0

    fd.close()
    env.close()


RECORD = True
RECORDING_DIR = './recordings'


def test():
    if RECORD:
        env = gym.make(ENV_NAME, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env, RECORDING_DIR, episode_trigger=lambda x: x % 5 == 0)
    else:
        env = gym.make(ENV_NAME, render_mode=RENDER_MODE)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    action_low = env.action_space.low
    action_high = env.action_space.high

    agent = Agent(state_dim, action_dim, max_action, LR)
    agent.load('./saved', FILE + '_solved')

    for episode in range(1, 11):
        state = env.reset()
        state = state[0]

        for timestep in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            action = action + np.random.normal(0, EXPLORATION_NOISE, size=env.action_space.shape[0])
            action = action.clip(action_low, action_high)

            next_state, reward, done, _, _ = env.step(action)
            state = next_state

            if done or timestep == (MAX_STEPS_PER_EPISODE - 1):
                break

    env.close()


IS_TRAINING = True
IS_TESTING = False

if __name__ == '__main__':
    if IS_TRAINING:
        train()
    if IS_TESTING:
        test()
