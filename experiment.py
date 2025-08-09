# let's experiments :)

from agents.DDPG_Agent import Agent
import gymnasium as gym 
import numpy as np
from utils.utils import plotLearning

# initiate env
env = gym.make("LunarLander-v3", continuous=True) 
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env, 
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = []

for i in range(1000): # 1000 eps
    done = False 
    score = 0
    obs = env.reset()
    while not done: 
        act = agent.choose_action(obs) # agent choose action based on current state
        new_state, reward, done, info = env.step(act) # apply action to get next state, reward, and ep status
        agent.remember(obs, act, reward, new_state, int(done)) 
        agent.learn()
        score += reward 
        obs = new_state

    score_history.append(score)
    print(f"Episode: {i} | Score: {round(score, 2)} | Mean Average of 100 games: {np.mean(score_history[-100:])}")


    if i % 25 == 0: 
        agent.save_models()
    
filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)