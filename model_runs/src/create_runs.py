import os
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime as dt
from copy import deepcopy


# Maps server hostnames to a number of models to run in parallel
# Special hostname ":" corresponds to your local machine
servers = {
    # Enter server info as follows:
    # "<server>": <N CPUs>,
    ":": 2
}

config_dir = "08_06_2020"

if __name__ == "__main__":

    rng = np.random
    rng.seed(1111)
    main_dir = Path("model_runs/parallel_runs/")

    # Create a directory for the runs
    try:
        shutil.rmtree(main_dir)
    except Exception as E:
        E
    main_dir.mkdir(exist_ok=True)

    # Create the parameters:
    parameters_dict = dict()
    for file in os.listdir("config_files/" + config_dir):
        if ".config" in file:
            # Read the file
            with open(Path("config_files", config_dir, file), "r") as text_file:
                contents = text_file.read().split("\n")

                min_r0 = float(contents[0].split("=")[1])
                max_r0 = float(contents[1].split("=")[1])
                runs_per = int(contents[2].split("=")[1])
                n_agents = float(contents[3].split("=")[1])
                cm = float(contents[4].split("=")[1])
                daysinmodel = float(contents[5].split("=")[1])
                outputstr = contents[6].split("=")[1]

            # Update the doubling rate and R0
            with open(Path("model_runs", "covid", "orig_run", "parameters.json"), "r+") as jsonFile:
                data = json.load(jsonFile)

                data["base"]["limit_pop"] = int(n_agents)
                data["base"]["outputstr"] = outputstr

                data["base"]["time_limit"] = daysinmodel + 2

                data["covid19"]["r0_min"] = min_r0
                data["covid19"]["r0_max"] = max_r0

                data["covid19"]["initial_case_multiplier"] = cm

                parameters_dict[file] = data

    # Create the directories
    run_total = 0
    for file, parameters in parameters_dict.items():
        for i in range(runs_per):

            params = deepcopy(parameters)
            r0 = rng.uniform(params["covid19"]["r0_min"], params["covid19"]["r0_max"])
            params["covid19"]["r0"] = r0

            # Create the run directory
            run_dir = main_dir.joinpath("run_" + str(run_total))
            run_dir.mkdir(exist_ok=True)
            with open(run_dir.joinpath("parameters.json"), "w") as outfile:
                json.dump(params, outfile, indent=4)
            run_total += 1
