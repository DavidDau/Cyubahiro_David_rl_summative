"""
DQN Training Script
Kigali Urban Noise Inspection RL Environment
"""

import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import NoiseInspectionEnv



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



    model = DQN(

        policy="MlpPolicy",

        env=env,

        learning_rate=0.0005,

        buffer_size=50000,

        learning_starts=1000,

        batch_size=64,

        gamma=0.95,

        exploration_fraction=0.3,

        exploration_final_eps=0.05,

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