import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.ldm import Ldm
from src.report_results import ReportResults


@pytest.fixture(scope="function")
def model_function_scoped():
    """Use this fixture if you want a fresh copy of an LDM model to execute a test.

    Use instead of copy(model) inside of each test"""

    scenario = "src/tests"
    run = "parameters"
    model = Ldm(scenario=scenario, run=run)
    yield model
    rundir = Path(model.run_dir, "model_output")
    if rundir.exists():
        rundir.rmdir()


@pytest.fixture(scope="session")
def model():
    """This model is session scoped, which means its state will be tracked throughout test execution.

    You might use this for tests that check for attributes after model initialization that
    do not change. However, for tests that check changes in any state this is not advisable
    since you'll have to track the state of agents as well as the sequence of changes that occur"""

    scenario = "src/tests"
    run = "parameters"
    model = Ldm(scenario=scenario, run=run)
    yield model
    rundir = Path(model.run_dir, "model_output")
    if rundir.exists():
        rundir.rmdir()


@pytest.fixture(scope="session")
def multiple_run_results_regular(tmpdir_factory):
    """Run two iterations of the model for tests that
    want to check output
    NOTE: use r0=~2.5 for 30 days for reasonable result testing
    use r0=8 and 10 days to test functionallity (forces things to happen)
    """
    agg_dir = tmpdir_factory.mktemp("AGG_DIR")
    run_directory = os.path.join("model_runs", "test_runs")
    Path(run_directory).mkdir(parents=True, exist_ok=True)

    # Change these if needed at the beginning of the run
    parameters_file = os.path.join("model_runs", "covid", "orig_run", "parameters.json")
    scenario_name = "covid"
    run_name = "orig_run"
    df_filename = "run_output.csv"

    run_val = -1
    ranges = [[1.3, 1.99], [2, 3.001]]
    days = None
    for j in range(2):
        for i in range(2):
            run_val += 1

            print("starting run " + str(i))

            # Sample R0 from distribution
            r0 = np.random.uniform(ranges[j][0], ranges[j][1])

            print("r0 is " + str(r0))

            results_directory = os.path.join(run_directory, f"run_{run_val}", "model_output")
            Path(results_directory).mkdir(parents=True, exist_ok=True)

            # Update the doubling rate and R0
            with open(parameters_file, "r+") as jsonFile:
                data = json.load(jsonFile)

                data["covid19"]["r0"] = r0
                days = data["base"]["time_horizon"]

                jsonFile.seek(0)  # rewind
                json.dump(data, jsonFile, indent=4)
                jsonFile.truncate()

                r_min = str(data["covid19"]["r0_min"]).replace(".", "p")
                r_max = str(data["covid19"]["r0_max"]).replace(".", "p")
                mult = f'x{int(data["covid19"]["initial_case_multiplier"])}'

                data["base"]["outputstr"] = f"'r{run_val}_{r_min}_{r_max}_{mult}_h?p?'"
                with open(
                    os.path.join(run_directory, f"run_{run_val}", "parameters.json"),
                    "w",
                ) as other_loc:
                    json.dump(data, other_loc, indent=4)

            # Run the model for 50 days, and save as a pickle
            ldm = Ldm(scenario=scenario_name, run=run_name)
            ldm.run_model()

            # Create export document
            export = ReportResults(ldm)

            r0_export = export.results_singular_run()
            r0_export.to_csv(os.path.join(results_directory, df_filename), index=False)

    os.system(f"python model_runs/src/aggregate_results.py --run_dir {run_directory} --output_dir {agg_dir}")

    shutil.rmtree(run_directory)

    return {"output_dir": agg_dir, "days": days}


@pytest.fixture()
def region_list():
    county_df = pd.read_csv("data/county_codes.csv")
    return {"regions": list(county_df["flu_regions"].astype(str).unique())}
