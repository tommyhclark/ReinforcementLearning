
import gymnasium as gym
from dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make("CarRacing-v3", render_mode=None, continuous=True)
agent = DQNAgent(input_shape=(3, 96, 96), num_actions=5)

num_episodes = 500
target_update_freq = 10  # Update target network every 10 episodes
reward_history = []

DISCRETE_ACTIONS = [
    np.array([0.0, 0.5, 0.0], dtype=np.float32),  # No action
    np.array([-1.0, 0.5, 0.0], dtype=np.float32), # Steer left
    np.array([1.0, 0.5, 0.0], dtype=np.float32),  # Steer right
    np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Accelerate
    np.array([0.0, 0.0, 0.1], dtype=np.float32)   # Brake
]


for episode in tqdm(range(num_episodes)):
    total_reward = 0
    obs, info = env.reset()
    obs = np.moveaxis(obs, -1, 0)

    for t in range(100):

        action = agent.select_action(obs)
        action_r = DISCRETE_ACTIONS[int(action)]
        next_obs, reward, terminated, truncated, info = env.step(action_r)
        next_obs = np.moveaxis(next_obs, -1, 0)

        agent.store_experience(obs, action, reward, next_obs, terminated)
        agent.train()

        obs = next_obs
        total_reward += reward
        if terminated or truncated:
            break
    
    reward_history.append(total_reward)  # Track rewards

    # Print progress
    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}")

env.close()



plt.figure(figsize=(10, 5))
plt.plot(range(len(reward_history)), reward_history, label="Episode Reward", color="blue", alpha=0.6)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Performance (CarRacing-v3)")
plt.legend()
plt.grid()
plt.show()


env.close()
