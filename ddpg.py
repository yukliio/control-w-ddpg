# source: https://spinningup.openai.com/en/latest/algorithms/ddpg.html, https://arxiv.org/pdf/1509.02971
# need replay buffer class
# need tagret q net 
# batch norm
# deterministic policy, handle explore exploit with mean-zero Gaussian noise
# target for actor and crtitic + target for each 
# soft updates according to theta_prime = tau*theta + (1-tau)*theta_prime, with tau << 1 to stabalize learning :) 


