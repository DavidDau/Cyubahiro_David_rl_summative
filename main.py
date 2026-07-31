"""
Main entry point for Kigali Urban Noise Inspection RL Project.
"""

from environment.custom_env import NoiseInspectionEnv


def run_environment():

    env = NoiseInspectionEnv(
        render_mode="human"
    )


    observation, info = env.reset()


    print("Environment Started")
    print("-------------------")
    print("Initial Observation:")
    print(observation)


    done = False


    while not done:

        action = env.action_space.sample()


        observation, reward, terminated, truncated, info = env.step(
            action
        )


        done = terminated or truncated


        print(
            f"Action: {action} | "
            f"Reward: {reward:.2f} | "
            f"Zone: {info['zone']} | "
            f"Violations: {info['violations']}"
        )


        env.render()



    print("\nMission Complete")

    print(
        "Total Violations Found:",
        info["violations"]
    )


    env.close()



if __name__ == "__main__":

    run_environment()