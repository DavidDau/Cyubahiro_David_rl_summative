"""
A2C Training Script
Kigali Urban Noise Inspection RL Environment
"""

import os

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import NoiseInspectionEnv



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



    model = A2C(

        policy="MlpPolicy",

        env=env,

        learning_rate=0.0007,

        n_steps=5,

        gamma=0.99,

        gae_lambda=1.0,

        ent_coef=0.01,

        vf_coef=0.5,

        max_grad_norm=0.5,

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