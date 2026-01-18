import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from dqn import DQN

class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Q-network & target network
        self.q_network = DQN(input_shape, num_actions).to(self.device)
        self.target_network = DQN(input_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Sync weights
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995  # Decay over time
        self.epsilon_min = 0.1  # Minimum exploration
        self.batch_size = 32
        self.memory = deque(maxlen=10000)  # Replay buffer

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 5)  # Explore (random action)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.q_network(state)).item()  # Exploit (best Q-value)

    def store_experience(self, state, action, reward, next_state, done):
        """Stores experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Samples from memory and trains the network"""
        if len(self.memory) < self.batch_size:
            return  # Skip if not enough experiences

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values from the main network
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using the target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and optimize
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from the Q-network to the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
