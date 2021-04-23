import os
import multiprocessing

from pathlib import Path


def run_scenario(run: str):
    os.system("python run_model.py parallel_runs " + run)


if __name__ == "__main__":
    """ Run all scenarios for the machine learning step
    """

    scenario_dir = Path("model_runs/parallel_runs")
    os.makedirs(scenario_dir, exist_ok=True)
    # ----- Run the scenarios ------------------------------------------------------------------------------------------
    core_count = int(multiprocessing.cpu_count() * 0.9) - 1
    if core_count < 1:
        core_count = 1
    with multiprocessing.Pool(core_count) as pool:
        tasks = []
        for run in [run_dir for run_dir in os.listdir(scenario_dir) if "run_" in run_dir]:
            run_dir = str(scenario_dir.joinpath(run))
            if "model_output" not in os.listdir(run_dir):
                tasks.append({"run": run, "result": pool.apply_async(func=run_scenario, kwds=dict(run=run))})
        pool.close()
        for task in tasks:
            try:
                task["result"].get()
            except Exception as E:
                print("Problem while running directory " + str(task["run"]) + ". Exception was: {}".format(E))
        pool.join()
