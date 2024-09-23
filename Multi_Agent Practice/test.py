import numpy as np
from pettingzoo.mpe import simple_adversary_v3
# from pettingzoo.test import render_test

import torch as T
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
print(device)


env = simple_adversary_v3.env(render_mode="human")
observation = env.reset(seed=42)
# render_test(env)

def print_details(env, observation):

    print('Number of agents: ', len(env.agents))  # Updated to get number of agents
    print('Observation space for the first agent: ', env.observation_space(env.agents[2]).shape)
    print('Action space for the first agent: ', env.action_space(env.agents[0]))
    print('Number of actions for the first agent: ', env.action_space(env.agents[0]).n)
    print('Observation: ', observation)

print_details(env, observation)

observation = env.reset()
print(observation)


for agent in env.agent_iter():
    # Get the last observation, reward, termination, truncation, and info for the current 
    observation, reward, termination, truncation, info = env.last()

    # Check if the agent's episode is over
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()