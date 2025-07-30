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
    # µ₀(st): Noisy action at time t
    # µ(st | θ_µₜ): Deterministic policy output (actor network prediction)
    # N: Noise sampled from a Gaussian distribution (e.g., Normal(0, σ²))


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

  


