import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Replay buffer
class MemoryBuffer():
    def __init__(self):
        self.log_probs = []
        self.rewards = []
        self.masks = []
        self.values = []
        self.states = []
        self.actions = []
        self.final_value = 0

    def storeMemory(self, action, obs_tensor, log_prob, reward, mask, value):
        self.log_probs.append(log_prob)
        self.rewards.append(torch.from_numpy(reward).float().to(device))
        self.masks.append(torch.from_numpy(mask).float().to(device))
        self.values.append(value)
        self.states.append(obs_tensor)
        self.actions.append(action)

    def storeFinalValue(self, final_value):
        self.final_value = final_value
        
    def getMemory(self):
        log_probs_tensor = torch.stack(self.log_probs).squeeze(-1).detach()
        values_tensor = torch.stack(self.values).squeeze(-1)
        rewards_tensor = torch.stack(self.rewards)
        masks_tensor = torch.stack(self.masks).detach()
        state_tensor = torch.stack(self.states).detach()
        action_tensor = torch.stack(self.actions).detach()
        ## HMM Need to get final value in here. Maybe store it?
        next_values_tensor = torch.zeros_like(values_tensor)
        next_values_tensor[:-1,:] = values_tensor[1:,:]
        next_values_tensor[-1,:] = self.final_value.squeeze(-1)
        return state_tensor, action_tensor, log_probs_tensor, rewards_tensor, values_tensor, next_values_tensor, masks_tensor


# --- Policy Network ---
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
    def __init__(self,state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
    def forward(self, x):
        return self.net(x)
    
def compute_gae(values, next_values, rewards, masks, GAMMA, LAMBDA, BATCH_SIZE, device):
    advantages = []
    gae_step = torch.zeros(BATCH_SIZE).to(device=device)
    for r,m,v,next_v in zip(reversed(rewards),reversed(masks), reversed(values), reversed(next_values)):
        delta = r+GAMMA*next_v-v
        gae_step = delta+GAMMA*LAMBDA*m*gae_step
        advantages.append(gae_step)
    return torch.stack(advantages[::-1])

def ppo_update(states, actions, advantages, returns, old_log_probs, policy_net, value_net, EPSILON, optimizer, value_optimizer):
    # 1. Flatten the inputs (T * BATCH_SIZE)
    # states shape: (T, B, state_dim) -> (T*B, state_dim)
    states = states.view(-1, states.size(-1))
    actions = actions.view(-1)
    old_log_probs = old_log_probs.view(-1)
    advantages = advantages.view(-1)
    returns = returns.view(-1)

    # 2. Get current policy distribution and values
    logits = policy_net(states)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    # entropy = dist.entropy().mean()
    
    current_values = value_net(states).squeeze()

    # 3. Calculate the PPO Ratio
    # ratio = exp(log(new) - log(old))
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 4. Clipped Surrogate Objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 5. Value Loss (MSE between critic prediction and actual returns)
    value_loss = F.mse_loss(current_values, returns)

    # 6. Optimization Step
    # You can combine these or keep them separate
    optimizer.zero_grad()
    value_optimizer.zero_grad()
    
    # Total loss: Policy + Value - Entropy (for exploration)
    total_loss = policy_loss + 0.5 * value_loss # - 0.01 * entropy
    
    total_loss.backward()
    optimizer.step()
    value_optimizer.step()  
    

# HMM Fix to look like returns calculation but also note why. Processing in batches

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
LAMBDA = 0.95
EPSILON = 0.1

# --- Initialize Environment ---
# RecordEpisodeStatistics is the "secret sauce" for vectorized monitoring
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
        # Send to memory buffer
        buffer = MemoryBuffer()

        for _ in range(STEPS_PER_ITER):
            obs_tensor = torch.from_numpy(obs).float().to(device)
            logits = policy_net(obs_tensor)
            value = value_net(obs_tensor)
            dist = Categorical(logits=logits)
            
            action = dist.sample()
            # Step requires numpy, so move back to CPU
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            mask = 1.0 - (terminated | truncated)

            buffer.storeMemory(action, obs_tensor, dist.log_prob(action), reward, mask, value)

            if "_episode" in info:
                finished_indices = np.where(info["_episode"])[0] # True if finished
                for idx in finished_indices:
                    length = info["episode"]["l"][idx]
                    all_durations.append(float(length))
            
            obs = next_obs

        final_value = value_net(torch.from_numpy(next_obs).float().to(device))
        buffer.storeFinalValue(final_value)

        state_tensor, action_tensor, log_probs_tensor, rewards_tensor, values_tensor, next_values_tensor, masks_tensor = buffer.getMemory()

        advantages = compute_gae(values_tensor, next_values_tensor, rewards_tensor, masks_tensor, GAMMA, LAMBDA, BATCH_SIZE, device)
        advantages = advantages.detach()
        # Compute returns
        returns = advantages + values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        for _ in range(10):
            ppo_update(state_tensor, action_tensor, advantages, returns, log_probs_tensor, policy_net, value_net, EPSILON, optimizer, value_optimizer)

        if len(all_durations) > 0:
            plot_durations(all_durations)



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
env = gym.make("CartPole-v1", render_mode="human")
for i in range(5):
    state, _ = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = policy_net(state_tensor)
            # Take the highest probability action for the demo
            action = torch.argmax(logits, dim=1).item()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated # or truncated
env.close()