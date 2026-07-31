"""
A2C Training Script
Kigali Urban Noise Inspection RL Environment
"""

import os

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import NoiseInspectionEnv
from utils.config_loader import load_config


MODEL_PATH = "models/a2c"
LOG_PATH = "logs/a2c"



def train_a2c():

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
    "configs/a2c.yaml" )

    model = A2C(

        policy="MlpPolicy",

        env=env,

        learning_rate=config["learning_rate"],

        n_steps=config["n_steps"],

        gamma=config["gamma"],

        gae_lambda=config["gae_lambda"],

        ent_coef=config["entropy_coefficient"],

        vf_coef=config["value_function_coefficient"],

        max_grad_norm=config["max_gradient_norm"],

        verbose=1

    )



    model.learn(

        total_timesteps=50000,

        callback=eval_callback

    )



    model.save(

        f"{MODEL_PATH}/noise_a2c"

    )


    print(
        "A2C training completed"
    )



if __name__ == "__main__":

    train_a2c()