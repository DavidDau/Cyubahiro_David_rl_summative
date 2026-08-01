"""
Evaluation metrics for RL agents.

Supports:
- PPO
- DQN
- A2C
- REINFORCE
"""

import numpy as np
import torch


def evaluate_agent(
    model,
    env,
    episodes=20,
    algorithm="PPO"
):
    """
    Evaluate a trained reinforcement learning agent.

    Returns:
        Dictionary containing evaluation metrics.
    """

    rewards = []

    violations = []

    completion_rates = []

    episode_lengths = []

    inspected_zones = []

    for _ in range(episodes):

        observation, _ = env.reset()

        terminated = False

        truncated = False

        total_reward = 0

        steps = 0

        while not (terminated or truncated):

            # ----------------------------------------
            # Stable-Baselines3 algorithms
            # ----------------------------------------

            if algorithm in ["PPO", "DQN", "A2C"]:

                action, _ = model.predict(
                    observation,
                    deterministic=True
                )

                if isinstance(action, np.ndarray):
                    action = action.item()

            # ----------------------------------------
            # Custom REINFORCE policy
            # ----------------------------------------

            elif algorithm == "REINFORCE":

                state = torch.FloatTensor(
                    observation
                )

                with torch.no_grad():

                    probabilities = model(state)

                    action = torch.argmax(
                        probabilities
                    ).item()

            else:

                raise ValueError(
                    f"Unknown algorithm: {algorithm}"
                )

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            steps += 1

        rewards.append(total_reward)

        violations.append(
            info["violations"]
        )

        completion_rates.append(
            float(env._mission_complete())
        )

        episode_lengths.append(
            steps
        )

        inspected_zones.append(
            env._number_inspected()
        )

    return {

        "average_reward": float(
            np.mean(rewards)
        ),

        "reward_std": float(
            np.std(rewards)
        ),

        "average_violations": float(
            np.mean(violations)
        ),

        "mission_completion_rate": float(
            np.mean(completion_rates)
        ),

        "average_episode_length": float(
            np.mean(episode_lengths)
        ),

        "average_inspected_zones": float(
            np.mean(inspected_zones)
        )
    }