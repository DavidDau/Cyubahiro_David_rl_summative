"""
Main execution file.

Runs:
- Environment initialization
- Trained RL agent loading
- Agent simulation
- Visualization
"""

import os
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
# Model paths
# --------------------------------------------------

MODEL_PATHS = {

    "PPO": "models/ppo/best_model",

    "DQN": "models/dqn/best_model",

    "A2C": "models/a2c/best_model",

    "REINFORCE": "models/reinforce/noise_reinforce.pth"

}


# --------------------------------------------------
# Load agent
# --------------------------------------------------

def load_agent(algorithm, env):

    if algorithm == "PPO":

        return PPO.load(MODEL_PATHS["PPO"])

    elif algorithm == "DQN":

        return DQN.load(MODEL_PATHS["DQN"])

    elif algorithm == "A2C":

        return A2C.load(MODEL_PATHS["A2C"])

    elif algorithm == "REINFORCE":

        state_size = env.observation_space.shape[0]

        action_size = env.action_space.n

        model = PolicyNetwork(
            state_size,
            action_size
        )

        model.load_state_dict(
            torch.load(
                MODEL_PATHS["REINFORCE"],
                map_location=torch.device("cpu")
            )
        )

        model.eval()

        return model

    else:

        raise ValueError("Unsupported algorithm")


# --------------------------------------------------
# Simulation
# --------------------------------------------------

def run_simulation(algorithm="PPO"):

    env = NoiseInspectionEnv()

    renderer = NoiseRenderer()

    observation, info = env.reset()

    model = load_agent(
        algorithm,
        env
    )

    done = False

    total_reward = 0

    print(f"Running {algorithm} agent")

    while not done:

        if algorithm == "REINFORCE":

            state = torch.FloatTensor(
                observation
            )

            with torch.no_grad():

                probabilities = model(state)

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

        done = terminated or truncated

        total_reward += reward

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

    print(f"Final reward: {total_reward}")

    renderer.close()


# --------------------------------------------------
# Entry
# --------------------------------------------------

if __name__ == "__main__":

    run_simulation(
        algorithm="REINFORCE"
    )