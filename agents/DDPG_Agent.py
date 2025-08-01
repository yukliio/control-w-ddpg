import os 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ReplayBuffer import ReplayBuffer as ReplayBuffer
from utils.OUActionNoise import OUActionNoise as OUActionNoise
from agents.ActorNetwork import ActorNetwork as ActorNetwork
from agents.CriticNetwork import CrticNetwork as CriticNetwork

# Agent -  manages the networks, replay buffer, and learning to interact with the enviornment for continuous action spaces.
class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300, 
                 batch_size=64):
        self.gamma = gamma # discount factor (emphesis on future rewards)
        self.tau = tau # sof update coefficient to stabilize training
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha=alpha, input_dims=input_dims, fc1_dims=layer1_size, 
                                fc2_dims=layer2_size, n_actions=n_actions, name='Actor')  # the policy network (what action to take)
        self.target_actor = ActorNetwork(alpha=alpha, input_dims=input_dims, fc1_dims=layer1_size, 
                                fc2_dims=layer2_size, n_actions=n_actions, name='TargetActor') # slowly upated copy of actor
        
        self.critic = CriticNetwork(beta=beta, input_dims=input_dims, fc1_dims=layer1_size, 
                                    fc2_dims=layer2_size, n_actions=n_actions, name='Critic') # Q-value computing network
        self.target_critic = CriticNetwork(beta=beta, input_dims=input_dims, fc1_dims=layer1_size, 
                                    fc2_dims=layer2_size, n_actions=n_actions, name='TargetCritic') # slowly updated copy critic
        
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.update_network_parameters(tau=1) # full copy of network parameters in the beginning


    # compute action given the current state (observation)
    def choose_action(self, observation): 
        self.actor.eval() # doesn't do anything for now since i'm using layernorm instead of batchnorm. will test batchnorm later on, though.
        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device) # puts input state to a tensor float + moves it to the same device as actor
        mu = self.actor(observation).to(self.actor.device) # the action output from actor network after pass given that state (deterministic)
        mu_prime = mu + torch.tensor(self.noise(), 
                                     dtype=torch.float).to(self.actor.device) # add noise to mu to encourage exploration
        self.actor.train()
        return mu_prime.cpu().detach().numpy() # moves mu_prime back to cpu and converts it to numpy array to pass into openai gym later
    

    # stores a transition into replay buffer to be sampled later when training
    def remember(self, state, action, reward, new_state, done): 
        self.memory.store_transition(state, action, reward, new_state, done)

    
    # (NOT COMPLETE) trains the actor and critic networks using a sampled mini-batch from the replay buffer
    def learn(self):
        # only starts learning if there are enough experiences in a batch
        if self.memory.mem_cntr < self.batch_size: 
            return 
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size) # sample a batch of transitions from replay buffer (state, action, reward, next_state, done)
        
        # each numpy array from the buffer is converted into a pytorch tensor, and sends it to the same device as the critic.
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done).to(self.critic.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)

        # set networks into evaluation mode (as i said, doesn't do anything...yet)
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

