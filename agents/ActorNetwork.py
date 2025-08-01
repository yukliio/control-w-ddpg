import os 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim


# ActorNetwork - use a neural net to map state to action.    
class ActorNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, 
                 chkpnt_dir='tmp/ddpg'): 