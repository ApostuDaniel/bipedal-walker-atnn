from datetime import datetime

import gymnasium as gym
# import pybullet
# import pybullet_envs
import numpy as np
from sac_torch import Agent
import matplotlib.pyplot as plt


def plot_learning(episodes, score_history, avg_score_history):
    plt.plot(episodes, score_history, color='b', label='Score')
    plt.plot(episodes, avg_score_history, color='r', label='Avg. score')
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()
    plt.savefig('plots/plot_' + datetime.now().strftime('%d-%m-%Y_%H_%M_%S') + '.png')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make("InvertedPendulum-v4", render_mode="human")
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    n_games = 500
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    avg_score_history = []
    load_chekcpoint = False

    if load_chekcpoint:
        env = gym.make("InvertedPendulum-v4", render_mode="human")
        agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
        agent.load_models()

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        truncated = False
        score = 0
        while not (done or truncated):
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done or truncated)
            if not load_chekcpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-(len(score_history)) if len(score_history) else -100])
        avg_score_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            if not load_chekcpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_chekcpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning(x, score_history, avg_score_history)
