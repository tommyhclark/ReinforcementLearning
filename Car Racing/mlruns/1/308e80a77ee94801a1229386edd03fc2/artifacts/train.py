import sys
import gymnasium as gym
import torch
import mlflow
import numpy as np
from typing import Any, Dict, Tuple, Union
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
import subprocess

# --- MLFLOW LOGGING INFRASTRUCTURE ---
class MLflowOutputFormat(KVWriter):
    """
    Custom Logger: Intercepts SB3 internal metrics and sends them to MLflow.
    """
    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and "mlflow" in excluded:
                continue
            if isinstance(value, np.ScalarType) and not isinstance(value, str):
                mlflow.log_metric(key, value, step=step)

def get_git_revision_hash():
    try:
        # Gets the short 7-character commit hash
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        return "no-git"

# --- MAIN TRAINING LOOP ---
def train():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. Start MLflow Experiment
    mlflow.set_experiment("CarRacing_Expert_Path")
    
    with mlflow.start_run(run_name="PPO_Baseline_SB3"):
        mlflow.log_artifact(__file__)
        git_hash = get_git_revision_hash()
        mlflow.set_tag("git_commit", git_hash)

        # Log Hyperparameters
        params = {"learning_rate": 1e-4, "n_steps": 5096, "batch_size": 64}
        mlflow.log_params(params)

        # 2. Setup Environment & Logger
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        
        # We tell SB3 to log to BOTH the terminal (HumanOutput) and MLflow
        loggers = Logger(
            folder=None, 
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()]
        )

        # 3. Initialize Model
        model = PPO("CnnPolicy", env, verbose=1, device=device, **params)
        model.set_logger(loggers)

        # 4. Train
        print(f"🚀 Training on {device}. Check MLflow UI for progress.")
        model.learn(total_timesteps=1e6)

        # 5. Save Artifacts
        model.save("ppo_car_racer")
        mlflow.log_artifact("ppo_car_racer.zip")
        print("✅ Training complete and model logged to MLflow.")

if __name__ == "__main__":
    train()