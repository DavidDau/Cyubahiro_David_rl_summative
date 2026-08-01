"""
PPO Training Script
Kigali Urban Noise Inspection RL Environment
"""

import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import NoiseInspectionEnv
from utils.config_loader import load_config


MODEL_PATH = "models/ppo"
LOG_PATH = "logs/ppo"



def train_ppo():

    os.makedirs(
        MODEL_PATH,
        exist_ok=True
    )

    os.makedirs(
        LOG_PATH,
        exist_ok=True
    )


    env = NoiseInspectionEnv()


    eval_env = NoiseInspectionEnv()



    eval_callback = EvalCallback(

        eval_env,

        best_model_save_path=MODEL_PATH,

        log_path=LOG_PATH,

        eval_freq=5000,

        deterministic=True,

        render=False

    )


    config = load_config(
    "configs/ppo.yaml"
    )
    model = PPO(

    policy="MlpPolicy",

    env=env,

    learning_rate=config["learning_rate"],

    n_steps=config["n_steps"],

    batch_size=config["batch_size"],

    n_epochs=config["n_epochs"],

    gamma=config["gamma"],

    gae_lambda=config["gae_lambda"],

    clip_range=config["clip_range"],

    ent_coef=config["entropy_coefficient"],

    verbose=1

)



    model.learn(

        total_timesteps=200000,

        callback=eval_callback

    )



    model.save(

        f"{MODEL_PATH}/noise_ppo"

    )


    print(
        "PPO training completed"
    )



if __name__ == "__main__":

    train_ppo()