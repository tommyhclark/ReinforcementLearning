import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import torch.nn.functional as F

gym.register(
    id="RocketLander-v0",
    entry_point="rocket_lander:RocketLander"
)

device = torch.device(
    #"cuda" if torch.cuda.is_available() else
    #"mps" if torch.backends.mps.is_available() else
    "cpu"
)

# --- Replay Buffer ---
class MemoryBuffer:
    def __init__(self):
        self.rewards = []
        self.values = []
        self.states = []
        self.actions = []
        self.masks = []
        self.log_probs = []
        self.final_value = 0

    def store_memory(self, action, obs_tensor, log_prob, reward, mask, value):
        self.log_probs.append(log_prob)
        self.rewards.append(torch.from_numpy(reward).float().to(device))
        self.masks.append(torch.from_numpy(mask).float().to(device))
        self.values.append(value)
        self.states.append(obs_tensor)
        self.actions.append(action)

    def store_final_value(self, final_value):
        self.final_value = final_value
        
    def get_memory(self):
        log_probs_tensor = torch.stack(self.log_probs).squeeze(-1).detach()
        values_tensor = torch.stack(self.values).squeeze(-1)
        rewards_tensor = torch.stack(self.rewards)
        masks_tensor = torch.stack(self.masks).detach()
        state_tensor = torch.stack(self.states).detach()
        action_tensor = torch.stack(self.actions).detach()
        
        # Handle final value for GAE calculation
        next_values_tensor = torch.zeros_like(values_tensor)
        next_values_tensor[:-1, :] = values_tensor[1:, :]
        next_values_tensor[-1, :] = self.final_value.squeeze(-1)
        
        return state_tensor, action_tensor, log_probs_tensor, rewards_tensor, values_tensor, next_values_tensor, masks_tensor


# --- Networks ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim)*-1)
    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- PPO Logic ---
def compute_gae(values, next_values, rewards, masks, gamma, lam, batch_size, device):
    advantages = []
    gae_step = torch.zeros(batch_size).to(device)
    for r, m, v, next_v in zip(reversed(rewards), reversed(masks), reversed(values), reversed(next_values)):
        delta = r + gamma * next_v - v
        gae_step = delta + gamma * lam * m * gae_step
        advantages.append(gae_step)
    return torch.stack(advantages[::-1])

def ppo_update(states, actions, advantages, returns, old_log_probs, policy_net, value_net, epsilon, optimizer, value_optimizer):
    # Flatten inputs for batch processing
    states = states.view(-1, states.size(-1))
    actions = actions.view(-1, actions.size(-1))
    old_log_probs = old_log_probs.view(-1)
    advantages = advantages.view(-1)
    returns = returns.view(-1)

    # Current policy and values
    mean, std = policy_net(states)
    dist = torch.distributions.Normal(mean, std)
    new_log_probs = dist.log_prob(actions).sum(dim=-1)
    current_values = value_net(states).squeeze()

    # PPO Ratio and Clipped Objective
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value Loss
    value_loss = F.mse_loss(current_values, returns)

    # Optimization
    optimizer.zero_grad()
    value_optimizer.zero_grad()
    
    total_loss = policy_loss + 0.5 * value_loss
    total_loss.backward()
    
    optimizer.step()
    value_optimizer.step()  

# --- Plotting ---
def plot_returns(all_returns):
    plt.figure(1)
    plt.clf()
    plt.title('Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(all_returns, color='royalblue', alpha=0.4, label='Raw')
    
    if len(all_returns) >= 50:
        means = torch.tensor(all_returns, dtype=torch.float).unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        plt.plot(means.numpy(), color='darkorange', linewidth=2, label='Smoothed')
    
    plt.legend()
    plt.draw()
    plt.pause(0.001)

# --- Hyperparameters ---
NUM_ITER = 20000
STEPS_PER_ITER = 300
BATCH_SIZE = 8
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.1

# --- Initialization ---
envs = gym.make_vec("RocketLander-v0", num_envs=BATCH_SIZE, vectorization_mode="sync")
envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

obs_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.shape[0]

action_low = torch.tensor(envs.single_action_space.low).to(device)
action_high = torch.tensor(envs.single_action_space.high).to(device)

policy_net = PolicyNetwork(obs_dim, action_dim).to(device)
value_net = ValueNetwork(obs_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

plt.ion()

def train():
    all_returns = []
    obs, _ = envs.reset()
    for i in range(NUM_ITER):
        buffer = MemoryBuffer()

        for _ in range(STEPS_PER_ITER):
            obs_tensor = torch.from_numpy(obs).float().to(device)
            
            with torch.no_grad():
                mean, std = policy_net(obs_tensor)
                value = value_net(obs_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                action_scaled = torch.clamp(action,action_low,action_high)
            
            next_obs, reward, terminated, truncated, info = envs.step(action_scaled.cpu().numpy())
            reward/=10
            mask = 1.0 - (terminated | truncated)

            buffer.store_memory(action, obs_tensor, log_prob, reward, mask, value)

            if "_episode" in info:
                finished_indices = np.where(info["_episode"])[0]
                for idx in finished_indices:
                    returns = info["episode"]["r"][idx]
                    all_returns.append(float(returns))
            
            obs = next_obs

        final_value = value_net(torch.from_numpy(next_obs).float().to(device))
        buffer.store_final_value(final_value)

        # Retrieve and calculate GAE
        states, actions, old_logs, rewards, values, next_values, masks = buffer.get_memory()
        advantages = compute_gae(values, next_values, rewards, masks, GAMMA, LAMBDA, BATCH_SIZE, device)
        advantages = torch.clamp(advantages, -5, 5)
        advantages = advantages.detach()

        
        returns = advantages + values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update loop
        for _ in range(10):
            ppo_update(states, actions, advantages, returns, old_logs, policy_net, value_net, EPSILON, optimizer, value_optimizer)

        if len(all_returns) > 0:
            plot_returns(all_returns)

# --- Execute ---
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        envs.close()

        # Visual Demo
        print("Launching Visual Demo...")
        demo_env = gym.make("RocketLander-v0", render_mode="human")
        for _ in range(5):
            obs, _ = demo_env.reset()
            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    mean, std = policy_net(obs_tensor)
                    dist = Normal(mean, std)
                    action = dist.sample()
                    action_scaled = torch.clamp(action,action_low,action_high)
                obs, _, terminated, truncated, _ = demo_env.step(action_scaled.cpu().numpy()[0])
                demo_env.render()
                done = terminated or truncated
        demo_env.close()
