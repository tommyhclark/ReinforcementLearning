import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

# Environment setup
env = gym.make("CartPole-v1", render_mode='human')
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

# DQN Definition
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Load pre-trained target network
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(torch.load('target_net.pth', weights_only=True, map_location=torch.device('cpu')))
target_net.eval()

# Action selection function
def select_action(state):
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    else:
        state = state.clone().detach()  # Safely handle tensor inputs
    return target_net(state).max(1).indices.view(-1, 1)

# Main loop
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Initial state as tensor
for i in range(2000):
    action = select_action(state)
    observation, reward, terminated, truncated, _ = env.step(action.item())  # Fixed line
    reward = torch.tensor([reward])
    done = terminated or truncated
    if done:
        print(f"Episode ended after {i+1} steps")
        break
    else:
        next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    state = next_state
    env.render()

env.close()