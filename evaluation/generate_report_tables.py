"""
Generate hyperparameter tables for report.
"""

import os
import json
import pandas as pd



INPUT_PATH = "logs/experiments"

OUTPUT_PATH = "assets/report_tables"



os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)



def save_table(
    algorithm,
    results
):

    rows = []


    for item in results:

        row = {

            "Run":
                item["run"],

            "Average Reward":
                round(
                    item["average_reward"],
                    2
                )

        }


        for key, value in item["hyperparameters"].items():

            row[key] = value


        rows.append(row)



    df = pd.DataFrame(rows)



    df.to_csv(

        f"{OUTPUT_PATH}/{algorithm}_hyperparameters.csv",

        index=False

    )


    print(
        f"{algorithm} table generated"
    )




def generate_all_tables():


    # DQN, PPO, A2C

    with open(

        f"{INPUT_PATH}/all_results.json",

        "r"

    ) as file:

        all_results = json.load(file)



    for algorithm, results in all_results.items():

        save_table(

            algorithm,

            results

        )



    # REINFORCE

    with open(

        f"{INPUT_PATH}/reinforce_results.json",

        "r"

    ) as file:

        reinforce_results = json.load(file)



    save_table(

        "REINFORCE",

        reinforce_results

    )




if __name__ == "__main__":

    generate_all_tables()