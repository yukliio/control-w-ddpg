# source: https://spinningup.openai.com/en/latest/algorithms/ddpg.html, https://arxiv.org/pdf/1509.02971
# need replay buffer class
# need tagrte q net 
# batch norm
# deterministic policy, handle explore exploit with mean-zero Gaussian noise
# target for actor and crtitic + target for each 
# soft updates according to theta_prime = tau*theta + (1-tau)*theta_prime, with tau << 1 to stabalize learning :) 

import os 
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

# µ₀(st) = µ(st | θ_µₜ) + N
# Where:
    # µ₀(st): noisy action at time t
    # µ(st | θ_µₜ): deterministic policy output (actor network prediction)
    # N: noise sampled from a Gaussian distribution (e.g., Normal(0, σ²))


class OUActionNoise(object): 
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None): 
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset() 

    def __call__(self): # this is so we can call this class like a regular function.
        # xₜ₊₁ = xₜ + θ(µ − xₜ) + σ * 𝒩(0, 1)
        # Where:
            # µ = mu (mean)
            # θ = theta (mean reversion speed)
            # σ = sigma (noise scale)
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
            # np.random.normal takes value from normal gaussian dist, and then creates an 
            # array of the same shape as self.mu, so the noise matches the diamentions of the 
            # action space. 
        self.x_prev = x 
        return x
    def reset(self): 
        # reset the noise after the start of a new ep to x0. sets the initial noise
        # value to either x0 (if given) or 0 (default), so the noise process has a starting point before evolving.
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

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
    

    
  


