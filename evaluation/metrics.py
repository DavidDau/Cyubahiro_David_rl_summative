"""
Evaluation metrics for RL agents.
"""

import numpy as np



def evaluate_agent(
    model,
    env,
    episodes=20
):
    """
    Evaluate trained RL agent performance.
    """

    rewards = []

    violations = []

    completion_rates = []

    episode_lengths = []



    for _ in range(episodes):

        obs, _ = env.reset()


        done = False

        total_reward = 0

        steps = 0



        while not done:


            action, _ = model.predict(
                obs,
                deterministic=True
            )

            if isinstance(action, np.ndarray):
                action = action.item()

            obs, reward, terminated, truncated, info = env.step(
                action
            )


            total_reward += reward

            steps += 1


            done = terminated or truncated



        rewards.append(
            total_reward
        )


        violations.append(
            info["violations"]
        )


        completion_rates.append(

            env._mission_complete()

        )


        episode_lengths.append(
            steps
        )



    return {

        "average_reward":
            np.mean(rewards),

        "reward_std":
            np.std(rewards),

        "average_violations":
            np.mean(violations),

        "mission_completion_rate":
            np.mean(completion_rates),

        "average_episode_length":
            np.mean(episode_lengths)

    }