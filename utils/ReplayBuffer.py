import os 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim


# ReplayBuffer - used to store memory of past states.
class ReplayBuffer(object): 
    def __init__(self, max_size, input_shape, n_actions): # max_size = max experiences to store, n_actions = number of possible actions
        self.mem_size = max_size # how many experiences we can store before overwriting 
        self.mem_cntr = 0 # a counter to track how many experiences have been stored (used for indexing, an int from 0 - mem_size)
        self.state_memory = np.zeros((self.mem_size, *input_shape)) # store past states — shape: (max_size, state_dim).
        self.new_state_memory = np.zeros((self.mem_size, *input_shape)) # next states
        self.action_memory = np.zeros((self.mem_size, n_actions)) # actions taken
        self.reward_memory = np.zeros(self.mem_size) # rewards after each action
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32) # stores done flags when entering terminal states
    def store_transition(self, state, action, reward, state_, done): 
        index = self.mem_cntr % self.mem_size # reset the index if mem_cntr > mem_size
        self.state_memory[index] = state 
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_ 
        self.terminal_memory[index] = 1 - done # using it in the bellman e to determine if we wanna use future Q vals
        self.mem_cntr +=1
    def sample_buffer(self, batch_size): 
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size) # batch_size indices from 0 to max_mem - 1

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal