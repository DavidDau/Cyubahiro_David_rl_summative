"""
Main execution file.

Runs:
- Environment initialization
- Trained RL agent loading
- Agent simulation
- Visualization
"""

import time
import os

from stable_baselines3 import PPO, DQN, A2C


from environment.custom_env import NoiseInspectionEnv

from environment.rendering import NoiseRenderer




MODEL_PATHS = {

    "PPO": "models/ppo/best_model",

    "DQN": "models/dqn/best_model",

    "A2C": "models/a2c/best_model"

}




def load_agent(
    algorithm="PPO"
):

    path = MODEL_PATHS[algorithm]


    if not os.path.exists(
        path + ".zip"
    ):

        raise FileNotFoundError(

            f"Model not found: {path}"

        )


    if algorithm == "PPO":

        model = PPO.load(
            path
        )


    elif algorithm == "DQN":

        model = DQN.load(
            path
        )


    elif algorithm == "A2C":

        model = A2C.load(
            path
        )


    else:

        raise ValueError(
            "Unsupported algorithm"
        )


    return model




def run_simulation(
    algorithm="PPO"
):


    env = NoiseInspectionEnv()



    model = load_agent(
        algorithm
    )



    renderer = NoiseRenderer()



    observation, info = env.reset()



    done = False



    total_reward = 0



    print(
        f"Running {algorithm} agent"
    )



    while not done:


        action, _ = model.predict(

            observation,

            deterministic=True

        )


        observation, reward, terminated, truncated, info = env.step(

            action

        )



        total_reward += reward



        done = terminated or truncated



        current_zone = env.current_zone



        renderer.draw_network(

            current_zone=current_zone

        )



        print(

            "Zone:",

            current_zone,

            "| Reward:",

            reward,

            "| Total:",

            total_reward

        )



        time.sleep(
            0.5
        )



    print(
        "Mission completed"
    )


    print(
        "Final reward:",
        total_reward
    )



    renderer.close()





if __name__ == "__main__":


    run_simulation(

        algorithm="PPO"

    )