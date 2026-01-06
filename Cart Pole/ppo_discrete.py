import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# --- Replay Buffer ---
class MemoryBuffer:
    def __init__(self):
        self.log_probs = []
        self.rewards = []
        self.masks = []
        self.values = []
        self.states = []
        self.actions = []
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
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)
    
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
    actions = actions.view(-1)
    old_log_probs = old_log_probs.view(-1)
    advantages = advantages.view(-1)
    returns = returns.view(-1)

    # Current policy and values
    logits = policy_net(states)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
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
def plot_durations(episode_durations):
    plt.figure(1)
    plt.clf()
    plt.title('Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations, color='royalblue', alpha=0.4, label='Raw')
    
    if len(episode_durations) >= 50:
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
LAMBDA = 0.95
EPSILON = 0.1

# --- Initialization ---
envs = gym.make_vec("CartPole-v1", num_envs=BATCH_SIZE, vectorization_mode="sync")
envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

policy_net = PolicyNetwork(4, 2).to(device)
value_net = ValueNetwork(4).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

plt.ion()

def train():
    all_durations = []
    obs, _ = envs.reset()
    for i in range(NUM_ITER):
        buffer = MemoryBuffer()

        for _ in range(STEPS_PER_ITER):
            obs_tensor = torch.from_numpy(obs).float().to(device)
            
            with torch.no_grad():
                logits = policy_net(obs_tensor)
                value = value_net(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
            
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            mask = 1.0 - (terminated | truncated)

            buffer.store_memory(action, obs_tensor, dist.log_prob(action), reward, mask, value)

            if "_episode" in info:
                finished_indices = np.where(info["_episode"])[0]
                for idx in finished_indices:
                    length = info["episode"]["l"][idx]
                    all_durations.append(float(length))
            
            obs = next_obs

        final_value = value_net(torch.from_numpy(next_obs).float().to(device))
        buffer.store_final_value(final_value)

        # Retrieve and calculate GAE
        states, actions, old_logs, rewards, values, next_values, masks = buffer.get_memory()
        advantages = compute_gae(values, next_values, rewards, masks, GAMMA, LAMBDA, BATCH_SIZE, device)
        advantages = advantages.detach()
        
        returns = advantages + values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update loop
        for _ in range(10):
            ppo_update(states, actions, advantages, returns, old_logs, policy_net, value_net, EPSILON, optimizer, value_optimizer)

        if len(all_durations) > 0:
            plot_durations(all_durations)

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
        demo_env = gym.make("CartPole-v1", render_mode="human")
        for _ in range(5):
            state, _ = demo_env.reset()
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits = policy_net(state_tensor)
                    action = torch.argmax(logits, dim=1).item()
                state, _, terminated, truncated, _ = demo_env.step(action)
                done = terminated or truncated
        demo_env.close()