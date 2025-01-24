import gymnasium as gym

import torch
import numpy as np
from soft_actor_critic.agent import Agent


if __name__ == '__main__':

    # load the environment
    env = gym.make('Pendulum-v1')

    # create the agent
    agent = Agent(
        alpha=0.0003,
        beta= 0.0003,
        input_dims= [8],
        hidden_dims= [256, 256],
        n_actions = 2,
        batch_size = 256,
        max_size = 1000000,
        tau = 0.005,
        env = env,
        gamma = 0.99,
        reward_scale = 2,
        dtype = torch.float32
    )

    # set the number of games and the best score
    n_games = 250
    best_score = float('-inf')

    # set the score history and the load checkpoint
    score_history = []
    load_checkpoint = False

    # load the checkpoint
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    # iterate over the number of games
    for i in range(n_games):

        # reset the environment
        observation = env.reset()
        done = False
        score = 0

        # iterate over the environment
        while not done:

            # sample actions from the agent
            action = agent.choose_action(observation) # TODO: convert the observation to a tensor
            # take a step in the environment
            observation_, reward, done, info = env.step(action) # TODO: convert the action to a numpy array
            # store the experience in the replay buffer
            agent.remember(observation, action, reward, observation_, done) # TODO: convert the observation, action, reward, observation_, and done to tensors

            # if the checkpoint is not loaded, learn
            if not load_checkpoint:
                agent.learn()

            # update the score and the observation
            score += reward
            observation = observation_

        # append the score to the score history
        score_history.append(score)
        # calculate the average score
        avg_score = np.mean(score_history[-100:])

        # if the average score is greater than the best score, update the best score
        if avg_score > best_score:

            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        # print the score and the average score
        print('episode', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)