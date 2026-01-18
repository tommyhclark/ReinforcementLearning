import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- End-to-End CNN Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self, action_dim=3, frame_stack=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(frame_stack * 3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(4096, 512)
        self.mu_head = nn.Linear(512, action_dim)
        self.sigma_head = nn.Linear(512, action_dim)

    def forward(self, x):
        # x shape for single env: (Stack, Height, Width, Channels) -> (4, 96, 96, 3)
        # We need to add a fake batch dimension for the CNN: (1, 4, 96, 96, 3)
        if x.ndimension() == 4:
            x = x.unsqueeze(0)
            
        b, s, h, w, c = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(b, s * c, h, w)
        x = x.float() / 255.0 
        x = self.cnn(x)
        x = torch.relu(self.fc(x))
        
        mu = torch.tanh(self.mu_head(x))
        sigma = torch.nn.functional.softplus(self.sigma_head(x)) + 1e-5
        return mu, sigma

# --- Environment Setup (Single Env) ---
def make_single_env(stack=4):
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    # Standard wrappers for a single environment
    env = gym.wrappers.FrameStackObservation(env, stack_size=stack)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

# --- Hyperparameters ---
GAMMA = 0.99
LR = 3e-4
NUM_ITER = 500
STEPS_PER_ITER = 1024 # Increased because we only have 1 env now

env = make_single_env()
policy_net = PolicyNetwork(action_dim=3).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# --- Training Loop ---
def train():
    obs, _ = env.reset()
    all_returns = []

    for i in range(NUM_ITER):
        log_probs, rewards, masks = [], [], []
        
        for _ in range(STEPS_PER_ITER):
            obs_tensor = torch.from_numpy(np.array(obs)).to(device)
            mu, sigma = policy_net(obs_tensor)
            dist = Normal(mu, sigma)
            
            action = dist.sample()
            # Clip for environment
            action_env = action.clone().cpu().numpy()[0] # Remove batch dim for single step
            action_env[1:] = np.clip(action_env[1:], 0.0, 1.0)
            action_env[0] = np.clip(action_env[0], -1.0, 1.0)

            next_obs, reward, terminated, truncated, info = env.step(action_env)
            
            log_probs.append(dist.log_prob(action).sum(dim=-1))
            rewards.append(torch.tensor([reward], device=device))
            masks.append(torch.tensor([1.0 - (terminated or truncated)], device=device))
            
            if terminated or truncated:
                if "episode" in info:
                    all_returns.append(info["episode"]["r"])
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Policy Update
        returns, R = [], torch.zeros(1).to(device)
        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + GAMMA * R * m
            returns.insert(0, R)
        
        returns = torch.cat(returns)
        log_probs = torch.cat(log_probs)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(log_probs * returns).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0 and all_returns:
            print(f"Iteration {i} | Avg Return: {np.mean(all_returns[-5:]):.2f}")

try:
    train()
finally:
    env.close()