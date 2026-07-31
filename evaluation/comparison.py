"""
Compare trained RL models:
DQN, PPO, A2C, REINFORCE
"""

import os

from stable_baselines3 import DQN, PPO, A2C

from environment.custom_env import NoiseInspectionEnv

from evaluation.metrics import evaluate_agent



MODEL_PATHS = {

    "DQN":
        "models/dqn/noise_dqn.zip",

    "PPO":
        "models/ppo/noise_ppo.zip",

    "A2C":
        "models/a2c/noise_a2c.zip"

}



def load_models():

    models = {}


    models["DQN"] = DQN.load(
        MODEL_PATHS["DQN"]
    )


    models["PPO"] = PPO.load(
        MODEL_PATHS["PPO"]
    )


    models["A2C"] = A2C.load(
        MODEL_PATHS["A2C"]
    )


    return models



def compare_models():


    env = NoiseInspectionEnv()


    models = load_models()


    results = {}



    for name, model in models.items():

        print(
            f"Evaluating {name}"
        )


        results[name] = evaluate_agent(

            model,

            env,

            episodes=20

        )



    print("\n===== MODEL COMPARISON =====")


    for model, metrics in results.items():

        print(
            f"\n{model}"
        )


        for metric, value in metrics.items():

            print(
                f"{metric}: {value:.3f}"
            )



    return results



if __name__ == "__main__":

    compare_models()