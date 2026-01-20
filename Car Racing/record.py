import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import mlflow
import os


# CONFIGURATION
MODEL_PATH = "ppo_car_racer.zip"
RUN_ID = "YOUR_RUN_ID_HERE"  # Copy the 'Run ID' from the MLflow UI

def record_and_log():
    # 1. Start the Environment
    def make_env():
        return gym.make("CarRacing-v3", render_mode="rgb_array")
    
    env = DummyVecEnv([make_env])
    
    # 2. Add Video Wrapper
    env = VecVideoRecorder(
        env, 
        video_folder="./videos",
        record_video_trigger=lambda x: x == 0, 
        video_length=1000,
        name_prefix="final_eval"
    )

    # 3. Load Model
    model = PPO.load(MODEL_PATH, env=env)

    # 4. Run & Record
    print("🎬 Recording the car...")
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
    
    env.close()

    # 5. Push to existing MLflow Run
    with mlflow.start_run(run_id=RUN_ID):
        video_path = "videos/final_eval-step-0-to-1000.mp4"
        if os.path.exists(video_path):
            mlflow.log_artifact(video_path, artifact_path="eval_videos")
            print(f"✅ Video logged to MLflow Run: {RUN_ID}")

if __name__ == "__main__":
    record_and_log()