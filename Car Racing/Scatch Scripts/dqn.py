import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        # Convolutional layers (feature extraction)
        # 32 learned filters being used, adapative to actual image size
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)  # 96x96 -> 23x23
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 23x23 -> 10x10
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 10x10 -> 8x8

        # Fully connected layers (decision making)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Flattened features -> 512 neurons
        self.fc2 = nn.Linear(512, num_actions)  # 512 neurons -> 5 actions (Q-values)

    def forward(self, x):
        x = x / 255.0  # Normalize pixel values to [0,1]
        x = F.relu(self.conv1(x))  # Apply first conv layer + ReLU
        x = F.relu(self.conv2(x))  # Apply second conv layer + ReLU
        x = F.relu(self.conv3(x))  # Apply third conv layer + ReLU

        x = x.view(x.size(0), -1)  # Flatten before feeding into FC layers
        x = F.relu(self.fc1(x))  # Fully connected layer + ReLU
        return self.fc2(x)  # Output Q-values for each action
