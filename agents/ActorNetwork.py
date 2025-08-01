import os 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim


# ActorNetwork - use a neural net to map state to action.    
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, 
                 chkpnt_dir='tmp/ddpg'): 
        super(ActorNetwork, self).__init__() # intiates everything from the parent class (nn.Module) 
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name 
        self.checkpoint_file = os.path.join(chkpnt_dir, name+'_ddpg')
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0]) # takes num of the output neurons of the layer. output neurons as sqrt args. weight should be scaled by 1 / sqrt(fan_int) to maintiann stable variance.
        nn.init.uniform_(self.fc1.weight.data, -f1, f1) # Fills the weight matrix with random vals between -f1 and f1.
        nn.init.uniform_(self.fc1.bias.data, -f1, f1) # same for bias
        

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0]) # takes num of the output neurons of the layer. output neurons as sqrt args. weight should be scaled by 1 / sqrt(fan_int) to maintiann stable variance.
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # define a forward pass 
    def forward(self, state, action): 
    
        return state_action_value
