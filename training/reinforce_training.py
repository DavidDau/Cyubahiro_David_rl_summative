"""
REINFORCE Policy Gradient Training Script
Kigali Urban Noise Inspection RL Environment
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.custom_env import NoiseInspectionEnv
from utils.config_loader import load_config


MODEL_PATH = "models/reinforce"



# ---------------------------------------------
# Policy Network
# ---------------------------------------------

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



# ---------------------------------------------
# Training
# ---------------------------------------------

config = load_config(
    "configs/reinforce.yaml"
)

def train_reinforce():

    os.makedirs(
        MODEL_PATH,
        exist_ok=True
    )


    env = NoiseInspectionEnv()


    state_size = env.observation_space.shape[0]

    action_size = env.action_space.n



    policy = PolicyNetwork(

        state_size,

        action_size

    )


    optimizer = optim.Adam(

        policy.parameters(),

         lr=config["learning_rate"]

    )



    episodes = config["episodes"]

    gamma = config["gamma"]



    for episode in range(episodes):


        state, _ = env.reset()



        rewards = []

        log_probs = []



        done = False



        while not done:


            state_tensor = torch.FloatTensor(
                state
            )


            probabilities = policy(
                state_tensor
            )


            distribution = torch.distributions.Categorical(
                probabilities
            )


            action = distribution.sample()



            log_prob = distribution.log_prob(
                action
            )


            next_state, reward, terminated, truncated, _ = env.step(

                action.item()

            )


            rewards.append(
                reward
            )


            log_probs.append(
                log_prob
            )


            state = next_state


            done = terminated or truncated



        # Calculate discounted returns

        returns = []

        discounted = 0



        for reward in reversed(rewards):

            discounted = reward + gamma * discounted

            returns.insert(
                0,
                discounted
            )



        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )


        returns = (
            returns - returns.mean()
        ) / (
            returns.std() + 1e-9
        )



        loss = 0



        for log_prob, reward_return in zip(

            log_probs,

            returns

        ):

            loss += (
                -log_prob * reward_return
            )



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        if episode % 50 == 0:

            total_reward = sum(rewards)


            print(

                f"Episode {episode} "
                f"Reward: {total_reward:.2f}"

            )



    torch.save(

        policy.state_dict(),

        f"{MODEL_PATH}/noise_reinforce.pth"

    )


    print(
        "REINFORCE training completed"
    )



if __name__ == "__main__":

    train_reinforce()