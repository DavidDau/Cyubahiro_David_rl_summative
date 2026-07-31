"""
Visualization and plotting utilities for RL evaluation.
"""

import os

import matplotlib.pyplot as plt

from evaluation.comparison import compare_models



PLOT_PATH = "assets/plots"



def create_plot_directory():

    os.makedirs(
        PLOT_PATH,
        exist_ok=True
    )



def plot_model_comparison(results):

    models = list(results.keys())


    rewards = [

        results[model]["average_reward"]

        for model in models

    ]


    violations = [

        results[model]["average_violations"]

        for model in models

    ]


    completion = [

        results[model]["mission_completion_rate"]

        for model in models

    ]



    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        models,
        rewards
    )


    plt.xlabel(
        "Algorithm"
    )

    plt.ylabel(
        "Average Reward"
    )


    plt.title(
        "RL Algorithm Reward Comparison"
    )


    plt.savefig(
        f"{PLOT_PATH}/reward_comparison.png"
    )


    plt.close()



    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        models,
        violations
    )


    plt.xlabel(
        "Algorithm"
    )


    plt.ylabel(
        "Detected Violations"
    )


    plt.title(
        "Noise Violation Detection Comparison"
    )


    plt.savefig(
        f"{PLOT_PATH}/violation_comparison.png"
    )


    plt.close()



    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        models,
        completion
    )


    plt.xlabel(
        "Algorithm"
    )


    plt.ylabel(
        "Completion Rate"
    )


    plt.title(
        "Mission Completion Comparison"
    )


    plt.savefig(
        f"{PLOT_PATH}/completion_comparison.png"
    )


    plt.close()



def generate_plots():

    create_plot_directory()


    results = compare_models()


    plot_model_comparison(
        results
    )


    print(
        "Evaluation plots generated"
    )



if __name__ == "__main__":

    generate_plots()