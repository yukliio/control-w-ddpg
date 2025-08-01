import os 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim


# CriticNetwork - use a neural net to compute Q given state-action pair.    
class CrticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, 
                 chkpnt_dir='tmp/ddpg'): 
        super(CrticNetwork, self).__init__() # intiates everything from the parent class (nn.Module) 
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims # 1st fully connected layer
        self.fc2_dims = fc2_dims # 2nd fully connected layer
        self.n_actions = n_actions
        self.name = name 
        self.checkpoint_file = os.path.join(chkpnt_dir, name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # input, output
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0]) # output neurons as sqrt args. weight should be scaled by 1 / sqrt(fan_int) to maintiann stable variance.
        nn.init.uniform_(self.fc1.weight.data, -f1, f1) # Fills the weight matrix with random vals between -f1 and f1.
        nn.init.uniform_(self.fc1.bias.data, -f1, f1) # same for bias
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0]) # output neurons as sqrt args. weight should be scaled by 1 / sqrt(fan_int) to maintiann stable variance.
        nn.init.uniform_(self.fc2.weight.data, -f2, f2) # Fills the weight matrix with random vals between -f1 and f1.
        nn.init.uniform_(self.fc2.bias.data, -f2, f2) # same for bias 
        self.bn1 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    # define a forward pass 
    def forward(self, state, action): 
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value) # produce final q value scalar
        return state_action_value
    

    # save the critic network’s parameters to a file specified by self.checkpoint_file.
    def save_checkpoint(self): 

        print("----saving checkpoint-----")
        torch.save(self.state_dict(), self.checkpoint_file)

    # load saved params from checkpoint into critic network to restore state
    def load_checkpoint(self): 
        print('---loading checkpoint-----')
        self.load_state_dict(torch.load(self.checkpoint_file))