"""
Generate report tables from RL experiment results.

Creates CSV tables for:
- DQN
- PPO
- A2C
- REINFORCE

Output:
assets/report_tables/
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



def generate_table(
    algorithm,
    results
):
    """
    Convert experiment results into CSV table.
    """


    rows = []


    # Handle dictionary results
    if isinstance(results, dict):

        results = [results]



    for index, item in enumerate(results):


        row = {

            "Algorithm":
                algorithm,

            "Run":
                index + 1

        }



        # Add metrics

        for key, value in item.items():

            if key != "hyperparameters":

                row[key] = value



        # Add hyperparameters if available

        if "hyperparameters" in item:


            for key, value in item["hyperparameters"].items():

                row[key] = value



        rows.append(row)



    df = pd.DataFrame(rows)



    output_file = (

        f"{OUTPUT_PATH}/"
        f"{algorithm}_hyperparameters.csv"

    )



    df.to_csv(

        output_file,

        index=False

    )



    print(
        f"Generated: {output_file}"
    )





def load_json(filename):


    path = os.path.join(

        INPUT_PATH,

        filename

    )


    if not os.path.exists(path):

        print(
            f"Skipping missing file: {path}"
        )

        return None



    with open(path, "r") as file:

        return json.load(file)






def generate_all_tables():


    # ------------------------------------
    # Stable Baselines algorithms
    # ------------------------------------

    all_results = load_json(
        "all_results.json"
    )


    if all_results:


        for algorithm, results in all_results.items():

            generate_table(

                algorithm,

                results

            )



    # ------------------------------------
    # REINFORCE
    # ------------------------------------

    reinforce_results = load_json(

        "reinforce_results.json"

    )


    if reinforce_results:


        generate_table(

            "REINFORCE",

            reinforce_results

        )



    print(
        "\nAll report tables generated successfully."
    )





if __name__ == "__main__":

    generate_all_tables()