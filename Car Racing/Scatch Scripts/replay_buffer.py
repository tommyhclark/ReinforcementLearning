import random 
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def store_experience(self, state, action, reward, next_state, done):
        """Stores a new experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size=32):
        """Returns a random batch of experiences for training."""
        batch = random.sample(self.memory, batch_size)
        # * unpacks [a,b,c]->a,b,c
        # zip takes [1,2,3],[a,b,c] and makes [1,a], [2,b] [3,c]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states), 
                np.array(actions), 
                np.array(rewards, dtype=np.float32), 
                np.array(next_states), 
                np.array(dones, dtype=np.float32))
    
    def size(self):
        """Returns the current size of the memory buffer."""
        return len(self.memory)