"""
Main execution file.

Runs:
- Environment initialization
- Trained RL agent loading
- Agent simulation
- Visualization
"""

import os
import sys
import time

import torch
import torch.nn as nn

from stable_baselines3 import PPO, DQN, A2C

from environment.custom_env import NoiseInspectionEnv
from environment.rendering import NoiseRenderer


# --------------------------------------------------
# REINFORCE Policy Network
# --------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Policy network used by the REINFORCE agent.
    """

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, action_size),

            nn.Softmax(dim=-1)

        )

    def forward(self, state):

        return self.network(state)


# --------------------------------------------------
# Model Paths
# --------------------------------------------------

MODEL_PATHS = {

    "PPO": "models/ppo/best_model",

    "DQN": "models/dqn/best_model",

    "A2C": "models/a2c/best_model",

    "REINFORCE": "models/reinforce/noise_reinforce.pth"

}


# --------------------------------------------------
# Load Agent
# --------------------------------------------------

def load_agent(
    algorithm,
    env
):
    """
    Load the requested trained agent.
    """

    algorithm = algorithm.upper()

    if algorithm == "PPO":

        path = MODEL_PATHS["PPO"] + ".zip"

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        return PPO.load(MODEL_PATHS["PPO"])


    elif algorithm == "DQN":

        path = MODEL_PATHS["DQN"] + ".zip"

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        return DQN.load(MODEL_PATHS["DQN"])


    elif algorithm == "A2C":

        path = MODEL_PATHS["A2C"] + ".zip"

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        return A2C.load(MODEL_PATHS["A2C"])


    elif algorithm == "REINFORCE":

        path = MODEL_PATHS["REINFORCE"]

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        state_size = env.observation_space.shape[0]

        action_size = env.action_space.n

        model = PolicyNetwork(

            state_size,

            action_size

        )

        model.load_state_dict(

            torch.load(

                path,

                map_location=torch.device("cpu")

            )

        )

        model.eval()

        return model


    else:

        raise ValueError(

            f"Unsupported algorithm: {algorithm}"

        )


# --------------------------------------------------
# Simulation
# --------------------------------------------------

def run_simulation(
    algorithm="PPO"
):
    """
    Execute one simulation episode using
    a trained reinforcement learning agent.
    """

    env = NoiseInspectionEnv()

    renderer = NoiseRenderer()

    observation, _ = env.reset()

    model = load_agent(

        algorithm,

        env

    )

    done = False

    total_reward = 0

    print(

        f"Running {algorithm} agent"

    )

    while not done:

        if algorithm.upper() == "REINFORCE":

            state = torch.FloatTensor(

                observation

            )

            with torch.no_grad():

                probabilities = model(

                    state

                )

            action = torch.argmax(

                probabilities

            ).item()

        else:

            action, _ = model.predict(

                observation,

                deterministic=True

            )

        observation, reward, terminated, truncated, info = env.step(

            action

        )

        total_reward += reward

        done = terminated or truncated

        renderer.draw_network(

            current_zone=env.current_zone

        )

        print(

            f"Zone: {env.current_zone} | "
            f"Reward: {reward} | "
            f"Total: {total_reward}"

        )

        time.sleep(0.5)

    print("\nMission completed")

    print(

        f"Final reward: {total_reward}"

    )

    renderer.close()


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":

    AVAILABLE_ALGORITHMS = {

        "PPO",

        "DQN",

        "A2C",

        "REINFORCE"

    }

    if len(sys.argv) < 2:

        print("\nUsage:")

        print("    python main.py <algorithm>\n")

        print("Available algorithms:")

        for algorithm in sorted(AVAILABLE_ALGORITHMS):

            print(f"    {algorithm}")

        sys.exit(0)

    algorithm = sys.argv[1].upper()

    if algorithm not in AVAILABLE_ALGORITHMS:

        print(f"\nUnknown algorithm: {algorithm}")

        print("\nAvailable algorithms:")

        for name in sorted(AVAILABLE_ALGORITHMS):

            print(f"    {name}")

        sys.exit(1)

    run_simulation(
        algorithm=algorithm
    )