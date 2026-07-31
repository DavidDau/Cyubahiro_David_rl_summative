"""
REINFORCE Hyperparameter Experiments

Runs 10 different configurations
and records performance.
"""

import os
import json

import torch
import torch.nn as nn
import torch.optim as optim

from environment.custom_env import NoiseInspectionEnv



RESULT_PATH = "logs/experiments"


os.makedirs(
    RESULT_PATH,
    exist_ok=True
)



EXPERIMENTS = [

    {
        "learning_rate":0.001,
        "gamma":0.90,
        "episodes":300
    },

    {
        "learning_rate":0.001,
        "gamma":0.95,
        "episodes":500
    },

    {
        "learning_rate":0.0005,
        "gamma":0.99,
        "episodes":500
    },

    {
        "learning_rate":0.0003,
        "gamma":0.99,
        "episodes":700
    },

    {
        "learning_rate":0.0001,
        "gamma":0.95,
        "episodes":500
    },

    {
        "learning_rate":0.0001,
        "gamma":0.999,
        "episodes":700
    },

    {
        "learning_rate":0.0007,
        "gamma":0.97,
        "episodes":400
    },

    {
        "learning_rate":0.0002,
        "gamma":0.98,
        "episodes":600
    },

    {
        "learning_rate":0.0008,
        "gamma":0.92,
        "episodes":300
    },

    {
        "learning_rate":0.00005,
        "gamma":0.99,
        "episodes":800
    }

]



class PolicyNetwork(nn.Module):


    def __init__(
        self,
        state_size,
        action_size
    ):

        super().__init__()


        self.model = nn.Sequential(

            nn.Linear(
                state_size,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                action_size
            ),

            nn.Softmax(
                dim=-1
            )

        )



    def forward(
        self,
        state
    ):

        return self.model(state)




def train_reinforce(config):


    env = NoiseInspectionEnv()



    policy = PolicyNetwork(

        env.observation_space.shape[0],

        env.action_space.n

    )



    optimizer = optim.Adam(

        policy.parameters(),

        lr=config["learning_rate"]

    )


    gamma = config["gamma"]



    for episode in range(
        config["episodes"]
    ):


        state, _ = env.reset()


        log_probs = []

        rewards = []


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


            log_probs.append(

                distribution.log_prob(action)

            )



            state, reward, terminated, truncated, _ = env.step(

                action.item()

            )


            rewards.append(
                reward
            )


            done = terminated or truncated



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


        loss = 0



        for log_prob, value in zip(

            log_probs,

            returns

        ):

            loss += (
                -log_prob * value
            )



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



    return policy




def evaluate(policy):


    env = NoiseInspectionEnv()


    scores = []



    for _ in range(10):


        state, _ = env.reset()


        done = False

        total_reward = 0



        while not done:


            state_tensor = torch.FloatTensor(
                state
            )


            probs = policy(
                state_tensor
            )


            action = torch.argmax(
                probs
            ).item()



            state, reward, terminated, truncated, _ = env.step(

                action

            )


            total_reward += reward


            done = terminated or truncated



        scores.append(
            total_reward
        )



    return sum(scores)/len(scores)




def run_experiments():


    results = []



    for index, config in enumerate(EXPERIMENTS):


        print(
            f"REINFORCE Run {index+1}/10"
        )


        policy = train_reinforce(
            config
        )


        score = evaluate(
            policy
        )



        results.append(

            {

                "run":
                    index+1,

                "hyperparameters":
                    config,

                "average_reward":
                    score

            }

        )



    with open(

        f"{RESULT_PATH}/reinforce_results.json",

        "w"

    ) as file:


        json.dump(

            results,

            file,

            indent=4

        )


    print(
        "REINFORCE experiments completed"
    )



if __name__ == "__main__":

    run_experiments()