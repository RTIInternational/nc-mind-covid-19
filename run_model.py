import argparse as arg
from pathlib import Path

from src.ldm import Ldm
from src.report_results import ReportResults


if __name__ == "__main__":
    description = """ Run a model. Must provide at least an experiemnt and a scenario directory."""

    parser = arg.ArgumentParser(description=description)

    parser.add_argument("scenario", default="location", help="The name of the scenario.")
    parser.add_argument("run", default="demo", help="The name of the run for the scenario")

    args = parser.parse_args()

    # initialize model
    ldm = Ldm(args.scenario, args.run)
    # Run the Model
    ldm.run_model()
    # Create export document
    days = ldm.params.get("base", {}).get("time_horizon", 0)
    if days > 0:
        export = ReportResults(ldm)
        r0_export = export.results_singular_run()
        r0_export["id"] = args.run
        r0_export.to_csv(Path(ldm.run_dir, "model_output", "run_output.csv"), index=False)
