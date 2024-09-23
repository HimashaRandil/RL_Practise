import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir, 
                 alpha = 0.01, beta = 0.01, fc1 = 64, fc2 = 64, gamma = 0.95, tau = 0.01):
        
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_idx = agent_idx
        self.agent_name = f'agent_{agent_idx}'



        # Actor and Critic for this agent 

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, chkpt_dir = chkpt_dir, name = self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions, chkpt_dir=chkpt_dir, name = self.agent_name+'_critic')


        # Target Actor and Critic
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                         chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,
                                           chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)  # Initialize with the same weights

    
    def choose_action(self, raw_obs):
        state = T.tensor([raw_obs], dtype=T.float).to(self.actor.device)
        action = self.actor.forward(state).cpu().detach().numpy()[0]
        return action

    

    '''
    def update_network_parameters(self, tau = None):
        
        if tau is None:
            tau = self.tau
        
        # Update target actor parameters

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() +\
                (1- tau) * target_actor_state_dict()    

        self.target_actor.load_state_dict(actor_state_dict)


        # Update target critic parameters

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                    (1 - tau) * target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        '''

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update target actor parameters
        target_actor_params = dict(self.target_actor.named_parameters())
        actor_params = dict(self.actor.named_parameters())

        for name in actor_params:
            actor_params[name] = tau * actor_params[name].clone() + \
                                (1 - tau) * target_actor_params[name].clone()

        self.target_actor.load_state_dict(actor_params)

        # Update target critic parameters
        target_critic_params = dict(self.target_critic.named_parameters())
        critic_params = dict(self.critic.named_parameters())

        for name in critic_params:
            critic_params[name] = tau * critic_params[name].clone() + \
                                (1 - tau) * target_critic_params[name].clone()

        self.target_critic.load_state_dict(critic_params)

            


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()


    def load_model(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()