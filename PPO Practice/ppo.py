import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
          self.states = []
          self.probs = []
          self.vals = []
          self.actions = []
          self.rewards = []
          self.truncated = []
          self.terminated = []

          self.batch_size = batch_size

    def generate_batches(self):
         n_states = len(self.states)
         batch_start = np.arange(0,n_states, self.batch_size)     
         # indices = np.arange(n_states, self.batch_size)
         indices = np.arange(n_states, dtype=np.int64)
         np.random.shuffle(indices)
         batches = [indices[i:i+self.batch_size] for i in batch_start]

         return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.truncated), \
                np.array(self.terminated), \
                batches
                
          
    def store_memory(self, state, action, probs, vals, reward, truncated, terminated):
         self.states.append(state)
         self.probs.append(probs)
         self.actions.append(action)
         self.vals.append(vals)
         self.rewards.append(reward)
         self.truncated.append(truncated)
         self.terminated.append(terminated)

        
    def clear_memory(self):
         self.states = []
         self.probs = []
         self.vals = []
         self.actions = []
         self.rewards = []
         self.truncated = []
         self.terminated = []


class ActorNetwork(nn.Module):
     def __init__(self, n_actions, input_dims, alpha, fc1_dims = 256, fc2_dims =256, chkpt_dir = 'tmp/ppo'):
          super(ActorNetwork, self).__init__()

          self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
          self.actor = nn.Sequential(
               nn.Linear(*input_dims, fc1_dims),
               nn.ReLU(), 
               nn.Linear(fc1_dims, fc2_dims),
               nn.ReLU(),
               nn.Linear(fc2_dims, n_actions),
               nn.Softmax(dim=-1)
          )

          self.optimizer = optim.Adam(self.parameters(), lr=alpha)
          self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
          self.to(self.device)


     def forward(self, state): 
          dist = self.actor(state)
          dist = Categorical(dist)

          return dist
     

     def save_checkpoint(self):
          T.save(self.state_dict(), self.checkpoint_file)


     def load_checkpoint(self):
          self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
     def __init__(self, input_dims, alpha, fc1_dims = 256, fc2_dims = 256, chkpt_dir = 'tmp/ppo'):
          super(CriticNetwork, self).__init__()

          self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
          self.critic = nn.Sequential(
               nn.Linear(*input_dims, fc1_dims),
               nn.ReLU(),
               nn.Linear(fc1_dims, fc2_dims),
               nn.ReLU(),
               nn.Linear(fc2_dims, 1)
          )

          self.optimizer = optim.Adam(self.parameters(), lr = alpha)
          self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
          self.to(self.device)


     def forward(self, state):
          value = self.critic(state)

          return value
     

     def save_checkpoint(self):
          T.save(self.state_dict(), self.checkpoint_file)


     def load_checkpoint(self):
          self.load_state_dict(T.load(self.checkpoint_file))
          

     
         
class Agent:
     def __init__ (self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                   policy_clip=0.2, batch_size=32, n_epochs=10):
          
          self.gamma = gamma
          self.policy_clip = policy_clip
          self.n_epochs = n_epochs
          self.gae_lambda = gae_lambda
          # self.bacth_size =  batch_size

          self.actor = ActorNetwork(n_actions, input_dims, alpha)
          self.critic = CriticNetwork(input_dims, alpha)
          self.memory = PPOMemory(batch_size)


     def remember(self, state, action, probs, vals, reward, truncated, terminated): 
          self.memory.store_memory(state, action, probs, vals, reward, truncated, terminated)
          # print('state in remember function', state)


     def save_models(self):
         print('...Saving Models...')
         self.actor.save_checkpoint()
         self.critic.save_checkpoint()


     def load_models(self):
          print('...Loading Models...')
          self.actor.load_checkpoint()
          self.critic.load_checkpoint()


     def choose_action(self, observation):
          # print('observation in choose_action function: ', observation)
          
          state = T.tensor([observation], dtype=T.float).to(self.actor.device)
          # print('state in choose_action function: ', state)

          dist = self.actor(state)
          value = self.critic(state)
          action = dist.sample()

          probs = T.squeeze(dist.log_prob(action)).item()
          action = T.squeeze(action).item()
          value = T.squeeze(value).item()

          return action, probs, value
     
    
     def learn(self):
          for _ in range(self.n_epochs):
               state_arr, action_arr, old_prob_arr, vals_arr,\
               reward_arr, truncated_arr, terminated_arr, batches =\
               self.memory.generate_batches()

               values = vals_arr
               dones_arr = np.logical_or(truncated_arr, terminated_arr)
               advantage = np.zeros(len(reward_arr), dtype=np.float32)
               
               # Generalized Advantage Estimation (GAE) algorithm.
               # t: the current time steps 
               # k: future time steps
               for t in range(len(reward_arr)-1): # iterate over all timesteps 
                    discount =1 
                    a_t = 0 
                    # calculates the advantage at each time step t by looking at future time steps (k).
                    '''
                    > reward_arr[k]: The reward received at step k.
                    > self.gamma * values[k + 1] * (1 - int(dones_arr[k])): The discounted value of the next state (i.e., what the agent expects for future rewards). 
                      It is set to 0 if the episode is done (dones_arr[k] is true).
                    > values[k]: The value estimate of the current state.
                    '''
                    for k in range(t, len(reward_arr) - 1): 
                         # Immediate reward and value adjustment
                         a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                                            (1-int(dones_arr[k])) - values[k])
                         discount *= self.gamma*self.gae_lambda
                    advantage[t] = a_t 
               
               advantage = T.tensor(advantage).to(self.actor.device)
               # Normalize the advantage
               # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
               # advantage = T.tensor(advantage).to(self.actor.device)

               values = T.tensor(values).to(self.actor.device)

               for batch in batches:
                    states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                    old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                    actions = T.tensor(action_arr[batch]).to (self.actor.device)

                    dist =  self.actor(states)
                    critic_value = self.critic(states)

                    critic_value = T.squeeze(critic_value)

                    new_probs = dist.log_prob(actions)
                    prob_ratio = new_probs.exp()/old_probs.exp()
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                                                     1+self.policy_clip)*advantage[batch]
                    actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                    returns = advantage[batch] + values[batch]
                    critic_loss = (returns - critic_value)**2
                    critic_loss = critic_loss.mean()

                    total_loss = actor_loss + 0.5*critic_loss

                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
                    total_loss.backward()

                    # Apply gradient clipping
                    T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                    T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                    self.actor.optimizer.step()
                    self.critic.optimizer.step()

          self.memory.clear_memory() 



