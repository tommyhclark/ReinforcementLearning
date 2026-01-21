import sys
import gymnasium as gym
import torch
import mlflow
import numpy as np
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger

# --- SCHEDULER ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# --- MLFLOW LOGGER (Unchanged) ---
class MLflowOutputFormat(KVWriter):
    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and "mlflow" in excluded:
                continue
            if isinstance(value, np.ScalarType) and not isinstance(value, str):
                mlflow.log_metric(key, value, step=step)

def train():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mlflow.set_experiment("CarRacing_Optimized")
    
    with mlflow.start_run(run_name="PPO_VecEnv_Scheduled"):
        # 1. Hyperparameters
        total_timesteps = 1_000_000
        params = {
            "n_steps": 2048,
            "batch_size": 128,
            "gamma": 0.99,
            "learning_rate": linear_schedule(3e-4), # Lowering LR over time
            "ent_coef": 0.01,                       # Keeps exploration alive
            "clip_range": 0.2,
            "n_epochs": 10,
        }
        mlflow.log_params({"lr_init": 3e-4, "total_timesteps": total_timesteps, **params})

        # 2. Setup Vectorized Environment
        # We use 4 parallel environments for faster data collection
        # GrayScale observation reduces complexity
        env = make_vec_env(
            "CarRacing-v3", 
            n_envs=4, 
            wrapper_class=gym.wrappers.GrayScaleObservation,
            wrapper_kwargs={"keep_dim": True}
        )
        
        # Stack 4 frames so the model sees motion
        env = VecFrameStack(env, n_stack=4)

        # 3. Callbacks for Early Stopping
        # This will stop training if the agent hits a mean reward of 900
        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
        eval_callback = EvalCallback(
            env, 
            callback_on_new_best=stop_callback,
            eval_freq=10000, 
            best_model_save_path="./logs/best_model",
            verbose=1
        )

        # 4. Logger Setup
        loggers = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()]
        )

        # 5. Model
        model = PPO("CnnPolicy", env, verbose=1, device=device, **params)
        model.set_logger(loggers)

        # 6. Train
        print(f"🚀 Training with FrameStack and Linear Scheduler on {device}...")
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        # 7. Save
        model.save("ppo_car_racer_optimized")
        mlflow.log_artifact("ppo_car_racer_optimized.zip")
        print("✅ Training complete.")

if __name__ == "__main__":
    train()