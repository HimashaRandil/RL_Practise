import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

from pettingzoo.mpe import simple_adversary_v3

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0 
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims

        # Initialize memory for the critic
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype = bool)

        self.init_actor_memory()

    
    def init_actor_memory(self):
        self.actor_state_memory = [] 
        self.actor_new_state_memory = []
        self.actor_action_memory = []


        for agent_idx in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size,self.actor_dims[agent_idx])))
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[agent_idx])))
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions)))


    def store_transition(self, actor_states, state, actions, reward, actor_states_, state_, done):
        # Circular Buffer
        index = self.mem_cntr % self.mem_size # Replace oldest if buffer is full

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = actor_states[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = actor_states_[agent_idx]
            self.actor_action_memory[agent_idx][index] = actions[agent_idx]


        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1 


    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch_indices = np.random.chhoice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch_indices]
        states_ = self.new_state_memory[batch_indices]
        rewards = self.reward_memory[batch_indices] 
        terminal = self.terminal_memory[batch_indices]

        actor_states = []
        actor_new_states = [] 
        actions = []

        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch_indices])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch_indices])  
            actions.append(self.actor_action_memory[agent_idx][batch_indices])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal    
    

    def ready(self):
        return self.mem_cntr >= self.batch_size