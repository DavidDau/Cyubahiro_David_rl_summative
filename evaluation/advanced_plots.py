"""
Advanced visualization tools for RL model comparison.

Generates:
- Reward comparison
- Mission completion comparison
- Violation detection comparison
- Algorithm summary plots
"""


import os
import json
import numpy as np
import matplotlib.pyplot as plt



RESULT_PATH = "logs/experiments"

OUTPUT_PATH = "assets/plots"



os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)



def load_results():

    results = {}


    files = {

        "DQN":
            "all_results.json",

        "REINFORCE":
            "reinforce_results.json"

    }


    # Load combined Stable Baselines results

    path = os.path.join(
        RESULT_PATH,
        "all_results.json"
    )


    if os.path.exists(path):

        with open(path, "r") as file:

            data = json.load(file)


            for algorithm, values in data.items():

                results[algorithm] = values



    # Load REINFORCE

    path = os.path.join(

        RESULT_PATH,

        "reinforce_results.json"

    )


    if os.path.exists(path):

        with open(path, "r") as file:

            results["REINFORCE"] = json.load(file)



    return results





def extract_metrics(results):

    metrics = {}


    for algorithm, runs in results.items():


        if isinstance(runs, dict):

            runs = [runs]



        rewards = []

        violations = []

        completion = []



        for run in runs:


            rewards.append(

                run.get(
                    "average_reward",
                    0
                )

            )


            violations.append(

                run.get(
                    "average_violations",
                    0
                )

            )


            completion.append(

                run.get(
                    "mission_completion_rate",
                    0
                )

            )



        metrics[algorithm] = {


            "reward":

                np.mean(rewards),


            "violations":

                np.mean(violations),


            "completion":

                np.mean(completion)

        }


    return metrics





def save_bar_chart(
    values,
    title,
    ylabel,
    filename
):


    algorithms = list(values.keys())

    scores = list(values.values())



    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        algorithms,
        scores
    )


    plt.title(
        title
    )


    plt.ylabel(
        ylabel
    )


    plt.xticks(
        rotation=45
    )


    plt.tight_layout()



    plt.savefig(

        f"{OUTPUT_PATH}/{filename}",

        dpi=300

    )


    plt.close()



def generate_plots():

    results = load_results()


    if not results:

        print(
            "No evaluation results found."
        )

        return



    metrics = extract_metrics(results)



    # ------------------------------
    # Reward comparison
    # ------------------------------

    save_bar_chart(

        {
            k:v["reward"]

            for k,v in metrics.items()

        },

        "Average Reward Comparison",

        "Average Reward",

        "reward_comparison.png"

    )



    # ------------------------------
    # Completion comparison
    # ------------------------------

    save_bar_chart(

        {
            k:v["completion"]

            for k,v in metrics.items()

        },

        "Mission Completion Rate",

        "Completion Rate",

        "completion_comparison.png"

    )



    # ------------------------------
    # Violation comparison
    # ------------------------------

    save_bar_chart(

        {
            k:v["violations"]

            for k,v in metrics.items()

        },

        "Noise Violation Detection",

        "Average Violations Found",

        "violation_comparison.png"

    )



    # ------------------------------
    # Overall comparison
    # ------------------------------

    overall = {}

    for algorithm, value in metrics.items():

        overall[algorithm] = (

            value["reward"]

            +

            value["violations"] * 50

        )



    save_bar_chart(

        overall,

        "Overall RL Algorithm Performance",

        "Performance Score",

        "algorithm_comparison.png"

    )



    print(
        "Advanced plots generated successfully."
    )





if __name__ == "__main__":

    generate_plots()