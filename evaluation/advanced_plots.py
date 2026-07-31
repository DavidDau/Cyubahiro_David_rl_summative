"""
Advanced visualization plots for RL experiments.

Generates:
- Reward comparison
- Hyperparameter convergence curves
- Algorithm performance comparison
"""

import os
import json

import matplotlib.pyplot as plt



INPUT_PATH = "logs/experiments"

OUTPUT_PATH = "assets/plots"



os.makedirs(

    OUTPUT_PATH,

    exist_ok=True

)



def load_results():

    with open(

        f"{INPUT_PATH}/all_results.json",

        "r"

    ) as file:

        algorithm_results = json.load(file)



    with open(

        f"{INPUT_PATH}/reinforce_results.json",

        "r"

    ) as file:

        algorithm_results["REINFORCE"] = json.load(file)



    return algorithm_results




def plot_reward_curves(results):


    plt.figure(
        figsize=(10,6)
    )



    for algorithm, runs in results.items():


        rewards = [

            run["average_reward"]

            for run in runs

        ]


        experiments = range(

            1,

            len(rewards)+1

        )



        plt.plot(

            experiments,

            rewards,

            marker="o",

            label=algorithm

        )



    plt.xlabel(
        "Experiment Run"
    )


    plt.ylabel(
        "Average Reward"
    )


    plt.title(
        "RL Hyperparameter Experiment Reward Curves"
    )


    plt.legend()


    plt.grid()


    plt.savefig(

        f"{OUTPUT_PATH}/reward_curves.png"

    )


    plt.close()




def plot_algorithm_comparison(results):


    algorithms = []

    averages = []



    for algorithm, runs in results.items():


        algorithms.append(
            algorithm
        )


        reward = sum(

            run["average_reward"]

            for run in runs

        ) / len(runs)



        averages.append(
            reward
        )



    plt.figure(

        figsize=(8,5)

    )



    plt.bar(

        algorithms,

        averages

    )



    plt.xlabel(
        "Algorithm"
    )


    plt.ylabel(
        "Average Reward"
    )


    plt.title(
        "Final RL Algorithm Comparison"
    )


    plt.grid(
        axis="y"
    )



    plt.savefig(

        f"{OUTPUT_PATH}/algorithm_comparison.png"

    )


    plt.close()




def plot_convergence(results):


    plt.figure(

        figsize=(10,6)

    )



    for algorithm, runs in results.items():


        rewards = [

            run["average_reward"]

            for run in runs

        ]


        cumulative = []


        total = 0



        for reward in rewards:


            total += reward

            cumulative.append(total)



        plt.plot(

            cumulative,

            label=algorithm

        )



    plt.xlabel(
        "Experiment"
    )


    plt.ylabel(
        "Cumulative Reward"
    )


    plt.title(
        "RL Training Convergence"
    )


    plt.legend()


    plt.grid()



    plt.savefig(

        f"{OUTPUT_PATH}/convergence_plot.png"

    )


    plt.close()




def generate_all_plots():


    results = load_results()



    plot_reward_curves(

        results

    )


    plot_algorithm_comparison(

        results

    )


    plot_convergence(

        results

    )



    print(
        "Advanced plots generated"
    )




if __name__ == "__main__":

    generate_all_plots()