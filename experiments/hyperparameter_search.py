"""
Hyperparameter Experiment Runner

Runs 10 experiments for:
- DQN
- PPO
- A2C
- REINFORCE

Stores results for analysis.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import DQN, PPO, A2C

from environment.custom_env import NoiseInspectionEnv



RESULT_PATH = "logs/experiments"


os.makedirs(
    RESULT_PATH,
    exist_ok=True
)



EXPERIMENTS = {


"DQN": [

    {"learning_rate":0.001,"gamma":0.90,"buffer_size":50000},
    {"learning_rate":0.001,"gamma":0.95,"buffer_size":100000},
    {"learning_rate":0.0005,"gamma":0.95,"buffer_size":50000},
    {"learning_rate":0.0005,"gamma":0.99,"buffer_size":100000},
    {"learning_rate":0.0003,"gamma":0.90,"buffer_size":50000},
    {"learning_rate":0.0003,"gamma":0.99,"buffer_size":200000},
    {"learning_rate":0.0001,"gamma":0.95,"buffer_size":100000},
    {"learning_rate":0.0001,"gamma":0.99,"buffer_size":200000},
    {"learning_rate":0.0007,"gamma":0.97,"buffer_size":50000},
    {"learning_rate":0.0002,"gamma":0.92,"buffer_size":100000}

],



"PPO": [

    {"learning_rate":0.001,"gamma":0.90,"clip_range":0.1},
    {"learning_rate":0.001,"gamma":0.95,"clip_range":0.2},
    {"learning_rate":0.0005,"gamma":0.99,"clip_range":0.2},
    {"learning_rate":0.0003,"gamma":0.99,"clip_range":0.3},
    {"learning_rate":0.0001,"gamma":0.95,"clip_range":0.1},
    {"learning_rate":0.0001,"gamma":0.999,"clip_range":0.3},
    {"learning_rate":0.0007,"gamma":0.97,"clip_range":0.2},
    {"learning_rate":0.0002,"gamma":0.98,"clip_range":0.15},
    {"learning_rate":0.0008,"gamma":0.92,"clip_range":0.25},
    {"learning_rate":0.00005,"gamma":0.99,"clip_range":0.1}

],



"A2C": [

    {"learning_rate":0.001,"gamma":0.90},
    {"learning_rate":0.001,"gamma":0.99},
    {"learning_rate":0.0007,"gamma":0.95},
    {"learning_rate":0.0007,"gamma":0.999},
    {"learning_rate":0.0005,"gamma":0.97},
    {"learning_rate":0.0003,"gamma":0.99},
    {"learning_rate":0.0002,"gamma":0.95},
    {"learning_rate":0.0001,"gamma":0.999},
    {"learning_rate":0.0008,"gamma":0.92},
    {"learning_rate":0.00005,"gamma":0.98}

]

}



def train_model(
    algorithm,
    params
):

    env = NoiseInspectionEnv()



    if algorithm == "DQN":

        model = DQN(

            "MlpPolicy",

            env,

            learning_rate=params["learning_rate"],

            gamma=params["gamma"],

            buffer_size=params["buffer_size"],

            verbose=0

        )



    elif algorithm == "PPO":

        model = PPO(

            "MlpPolicy",

            env,

            learning_rate=params["learning_rate"],

            gamma=params["gamma"],

            clip_range=params["clip_range"],

            verbose=0

        )



    elif algorithm == "A2C":

        model = A2C(

            "MlpPolicy",

            env,

            learning_rate=params["learning_rate"],

            gamma=params["gamma"],

            verbose=0

        )



    model.learn(
        total_timesteps=10000
    )


    return model




def evaluate_model(model):

    env = NoiseInspectionEnv()


    total_rewards = []


    for _ in range(10):


        obs, _ = env.reset()


        done = False

        reward_total = 0



        while not done:


            action, _ = model.predict(

                obs,

                deterministic=True

            )


            if hasattr(action,"item"):

                action = action.item()



            obs, reward, terminated, truncated, info = env.step(

                action

            )


            reward_total += reward


            done = terminated or truncated



        total_rewards.append(
            reward_total
        )



    return sum(total_rewards) / len(total_rewards)




def run_experiments():


    results = {}



    for algorithm, experiments in EXPERIMENTS.items():


        results[algorithm] = []



        for index, params in enumerate(experiments):


            print(
                f"{algorithm} Run {index+1}/10"
            )



            model = train_model(

                algorithm,

                params

            )


            score = evaluate_model(
                model
            )



            results[algorithm].append(

                {

                    "run":
                        index+1,

                    "hyperparameters":
                        params,

                    "average_reward":
                        score

                }

            )



    with open(

        f"{RESULT_PATH}/all_results.json",

        "w"

    ) as file:


        json.dump(

            results,

            file,

            indent=4

        )


    print(
        "All experiments completed"
    )



if __name__ == "__main__":

    run_experiments()