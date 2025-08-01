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

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300, 
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha=alpha, input_dims=input_dims, fc1_dims=layer1_size, 
                                fc2_dims=layer2_size, n_actions=n_actions, name='Actor')
        self.target_actor = ActorNetwork(alpha=alpha, input_dims=input_dims, fc1_dims=layer1_size, 
                                fc2_dims=layer2_size, n_actions=n_actions, name='TargetActor')
        
        self.critic = CriticNetwork(beta=beta, input_dims=input_dims, fc1_dims=layer1_size, 
                                    fc2_dims=layer2_size, n_actions=n_actions, name='Critic')
        self.target_critic = CriticNetwork(beta=beta, input_dims=input_dims, fc1_dims=layer1_size, 
                                    fc2_dims=layer2_size, n_actions=n_actions, name='TargetCritic')
        
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.update_network_parameters(tau=1)

    