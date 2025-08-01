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

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300, 
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)