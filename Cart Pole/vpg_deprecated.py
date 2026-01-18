import gymnasium as gym
import math
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from torch.distributions import Categorical
import time

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Create a policy network
class policyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,action_dim)
        )
    def forward(self, x):
        return self.net(x)

def getAction(state, net):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # Unsqueeze makes it a batch
    logits = net.forward(state_tensor)
    return torch.Categorical(logits=logits).sample().item()

def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    plt.pause(0.0001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


NUM_ITER = 200
BATCH_SIZE = 5
GAMMA = 0.99  # Discount factor
envs = gym.make_vec("CartPole-v1", num_envs=BATCH_SIZE, vectorization_mode="sync")
policy_net = policyNetwork(4,2).to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-2)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


durations = []
def train():
    for i in range(NUM_ITER):
        obs, _ = envs.reset()
        log_probs = []
        rewards = []
        masks = []
        durs = torch.zeros(BATCH_SIZE)
        batched_durs = []
        # Collect a fixed number of steps per iteration
        for _ in range(200):
            obs_tensor = torch.from_numpy(obs).float().to(device)
            logits = policy_net(obs_tensor)
            dist = Categorical(logits=logits)
            
            action = dist.sample()
            next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            
            log_probs.append(dist.log_prob(action))
            rewards.append(torch.from_numpy(reward).float().to(device))
            # Mask is 0 if episode ended, 1 otherwise
            masks.append(torch.from_numpy(1.0 - (terminated | truncated)).float().to(device))
            obs = next_obs
            durs[masks==1] += 1
            batched_durs.append(durs[masks==0])
            durs[masks==0] = 0
        durations.append(np.mean(batched_durs))

        # 3. Fast Vectorized Reward-to-go Calculation
        returns = []
        R = torch.zeros(BATCH_SIZE).to(device)
        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + GAMMA * R * m
            returns.insert(0, R)
        
        returns = torch.stack(returns)
        log_probs = torch.stack(log_probs)
        
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Loss and Update
        loss = -(log_probs * returns).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        plot_durations(durations)

train()
envs.close()

env = gym.make("CartPole-v1", render_mode="human")
for i in range(10):
    state, info = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy_net.forward(state_tensor)
        sampler = Categorical(logits=logits)
        action = sampler.sample()
        action = action.item()
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        env.render()
        time.sleep(0.01)
env.close()





# # Create a value network
# class valueNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(4,10),
#             nn.ReLU(),
#             nn.Linear(10,10),
#             nn.ReLU(),
#             nn.Linear(10,1)
#         )

#     def forward(self, x):
#         return self.network(x)

# def getValue(state, net):
#     state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # Unsqueeze makes it a batch
#     value = net.forward(state_tensor)
#     return value