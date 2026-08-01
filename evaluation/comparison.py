"""
Compare trained RL models:
- DQN
- PPO
- A2C
- REINFORCE
"""

import json
import os

import torch
from stable_baselines3 import DQN, PPO, A2C

from environment.custom_env import NoiseInspectionEnv
from evaluation.metrics import evaluate_agent
from training.reinforce_training import PolicyNetwork


MODEL_PATHS = {
    "DQN": "models/dqn/noise_dqn.zip",
    "PPO": "models/ppo/noise_ppo.zip",
    "A2C": "models/a2c/noise_a2c.zip",
    "REINFORCE": "models/reinforce/noise_reinforce.pth",
}

RESULT_PATH = "logs/experiments/all_results.json"


def load_models():
    """
    Load every trained model.
    """

    env = NoiseInspectionEnv()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    models = {}

    if os.path.exists(MODEL_PATHS["DQN"]):
        models["DQN"] = DQN.load(MODEL_PATHS["DQN"])

    if os.path.exists(MODEL_PATHS["PPO"]):
        models["PPO"] = PPO.load(MODEL_PATHS["PPO"])

    if os.path.exists(MODEL_PATHS["A2C"]):
        models["A2C"] = A2C.load(MODEL_PATHS["A2C"])

    if os.path.exists(MODEL_PATHS["REINFORCE"]):

        policy = PolicyNetwork(
            state_size,
            action_size
        )

        policy.load_state_dict(
            torch.load(
                MODEL_PATHS["REINFORCE"],
                map_location="cpu"
            )
        )

        policy.eval()

        models["REINFORCE"] = policy

    return models


def compare_models():

    env = NoiseInspectionEnv()

    models = load_models()

    results = {}

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")

    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        results[name] = evaluate_agent(
            model=model,
            env=env,
            episodes=20,
            algorithm=name
        )

    print("\n========== RESULTS ==========")

    for algorithm, metrics in results.items():

        print(f"\n{algorithm}")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

    os.makedirs(
        "logs/experiments",
        exist_ok=True
    )

    with open(
        RESULT_PATH,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(f"\nResults saved to {RESULT_PATH}")

    return results


if __name__ == "__main__":
    compare_models()