import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# --- Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(64, action_dim)
        self.sigma_head = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.base(x)
        mu = torch.tanh(self.mu_head(x))
        sigma = torch.nn.functional.softplus(self.sigma_head(x)) + 1e-5
        return mu, sigma

# --- Plotting Utility ---
def plot_durations(episode_durations):
    plt.figure(1)
    plt.clf()
    plt.title('Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations, color='royalblue', alpha=0.4, label='Raw')
    
    if len(episode_durations) >= 50:
        # Moving average to see through the noise
        means = torch.tensor(episode_durations, dtype=torch.float).unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        plt.plot(means.numpy(), color='darkorange', linewidth=2, label='Smoothed')
    
    plt.legend()
    plt.draw()
    plt.pause(0.001)

# --- Hyperparameters ---
NUM_ITER = 200
STEPS_PER_ITER = 1000
BATCH_SIZE = 8
GAMMA = 0.99

# --- Initialize Environment ---
# RecordEpisodeStatistics is the "secret sauce" for vectorized monitoring
envs = gym.make_vec("MountainCarContinuous-v0", num_envs=BATCH_SIZE, vectorization_mode="sync")
envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

policy_net = PolicyNetwork(2, 2).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

plt.ion()

def train():
    all_returns = []
    obs, _ = envs.reset()
    for i in range(NUM_ITER):
        log_probs = []
        rewards = []
        masks = []
        
        for _ in range(STEPS_PER_ITER):
            obs_tensor = torch.from_numpy(obs).float().to(device)
            mu, sigma = policy_net(obs_tensor)
            dist = torch.distributions.Normal(mu, sigma)
            
            
            action = dist.sample()
            # Clip the action to valid environment range [-1, 1]
            action_clipped = torch.clamp(action, -1.0, 1.0)
            # Step requires numpy, so move back to CPU
            next_obs, reward, terminated, truncated, info = envs.step(action_clipped.cpu().numpy())
            
            reward += abs(next_obs[:, 1])*100
            log_probs.append(dist.log_prob(action).sum(dim=-1))
            rewards.append(torch.from_numpy(reward).float().to(device))
            masks.append(torch.from_numpy(1.0 - (terminated | truncated)).float().to(device))
        
            if "_episode" in info:
                finished_indices = np.where(info["_episode"])[0] # True if finished
                for idx in finished_indices:
                    returned = info["episode"]["r"][idx]
                    all_returns.append(float(returned))
            
            obs = next_obs

        # Calculate Returns
        returns = []
        R = torch.zeros(BATCH_SIZE).to(device)
        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + GAMMA * R * m
            returns.insert(0, R)
        
        returns = torch.stack(returns)
        log_probs = torch.stack(log_probs)
        
        # Standardize returns (Standard REINFORCE trick)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Loss and Update
        loss = -(log_probs * returns).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if len(all_returns) > 0:
            plot_durations(all_returns)



# --- Execute ---
try:
    train()
except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
finally:
    envs.close()

# --- Visual Demo ---
print("Launching Visual Demo...")
# Re-create single env for rendering
env = gym.make("MountainCarContinuous-v0", render_mode="human")
for i in range(5):
    obs, _ = env.reset()
    done = False
    while not done:
        obs_tensor = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            mu, sigma = policy_net(obs_tensor)
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.sample()
            action_clipped = torch.clamp(action, -1.0, 1.0)
        obs, _, terminated, truncated, _ = env.step(action_clipped.cpu().numpy())
        done = terminated # or truncated
env.close()