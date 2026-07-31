"""
DQN Training Script
Kigali Urban Noise Inspection RL Environment
"""

import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import NoiseInspectionEnv
from utils.config_loader import load_config



MODEL_PATH = "models/dqn"
LOG_PATH = "logs/dqn"



def train_dqn():

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
        "configs/dqn.yaml"
    )

    model = DQN(

            policy="MlpPolicy",

            env=env,

            learning_rate=config["learning_rate"],

            buffer_size=config["buffer_size"],

            learning_starts=config["learning_starts"],

            batch_size=config["batch_size"],

            gamma=config["gamma"],

            exploration_fraction=config["exploration_fraction"],

            exploration_final_eps=config["exploration_final_eps"],

            verbose=1

        )



    model.learn(

        total_timesteps=50000,

        callback=eval_callback

    )



    model.save(

        f"{MODEL_PATH}/noise_dqn"

    )


    print(
        "DQN training completed"
    )



if __name__ == "__main__":

    train_dqn()