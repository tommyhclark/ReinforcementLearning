import os
import multiprocessing

# THIS MUST BE THE VERY FIRST PIECE OF CODE EXECUTED
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.envs.registration import register

# 1. Cleaner Registration
def register_env():
    try:
        register(
            id="RocketLander-v0",
            entry_point="rocket_lander.py",
            max_episode_steps=1000,
        )
    except Exception:
        # Avoid error if already registered in this session
        pass

def train():
    register_env()

    # Use 4 envs to start; 8 can sometimes saturate the Mac's efficiency cores
    n_envs = 4 
    
    print(f"--- Starting training with {n_envs} parallel environments ---")
    
    # Create the vectorized environment
    # Ensure start_method is passed to make_vec_env as well
    vec_env = make_vec_env(
        "RocketLander-v0", 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"}
    )

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
    model.learn(total_timesteps=100_000) # Increased for better results
    model.save("ppo_rocket_lander")
    print("Model saved.")
    
    vec_env.close()

def evaluate():
    # Never render in a SubprocVecEnv on Mac. Use a single env for visuals.
    print("--- Starting Evaluation ---")
    model = PPO.load("ppo_rocket_lander")
    env = gym.make("RocketLander-v0", render_mode="human")
    
    obs, _ = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

if __name__ == "__main__":
    # Choose action
    train()
    # evaluate()