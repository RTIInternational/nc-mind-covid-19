import argparse
import os
import pandas as pd
import numpy as np
import glob
import datetime

from collections import defaultdict
from src.parameters import load_parameters
from src.constants import POPULATION
from seir.src.simple_seir import prepare_data
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=""" Get results from the model.""")

    parser.add_argument(
        "--run_dir",
        default="model_runs/parallel_runs",
        help="The path to the directory where the run directories are located",
    )

    parser.add_argument(
        "--output_dir",
        default="parallel_reports/{}".format(str(datetime.datetime.today())),
        help="The path to the directory where aggregated results should be stored.",
    )
    args = parser.parse_args()

    params = load_parameters(Path(args.run_dir, "run_0", "parameters.json"))
    temp_df, _ = prepare_data(1)
    start_date = temp_df["North Carolina"].index[-1].date() + datetime.timedelta(1)
    end_date = pd.Timestamp(start_date + datetime.timedelta(params["base"]["time_horizon"]))

    output_dir = args.output_dir

    # Concatenate all output CSVs
    runs = defaultdict(list)
    for run in os.listdir(args.run_dir):
        if "run_" in run:
            params = load_parameters(Path(args.run_dir, run, "parameters.json"))
            runs[params["base"]["outputstr"].replace("'", "")].append(run)

    for output_name, files in runs.items():
        df_list = []
        for run in files:
            try:
                temp_df = pd.read_csv(Path(args.run_dir, run, "model_output/run_output.csv"))
                df_list.append(temp_df)
            except Exception as E:
                E
        df = pd.concat(df_list)

        # Set up output folders
        output_string = "output " + output_name + " " + str(datetime.datetime.today().date())
        Path(os.path.join(output_dir, output_string)).mkdir(parents=True, exist_ok=True)

        # Set dates
        df["date"] = pd.to_datetime(start_date)
        df["date"] = df["date"] + pd.TimedeltaIndex(df["day"], unit="D")
        df = df[df.date <= end_date]
        df = df[pd.Timestamp(start_date) < df.date]

        # calculate the hospitalization rate by run per 100,000 people
        covid_hospitalized_rate = df[["covid_seeking_hospital", "run_id"]].groupby("run_id").sum().astype(int)
        covid_infections = df[["newinfected", "run_id"]].groupby("run_id").sum().astype(int)
        covid_hospitalized_rate_summary = pd.DataFrame.from_dict(
            {
                "run": [output_name],
                "mean": [(covid_hospitalized_rate.mean().values * 100000) / POPULATION],
                "min": [(covid_hospitalized_rate.min().values * 100000) / POPULATION],
                "max": [(covid_hospitalized_rate.max().values * 100000) / POPULATION],
                "%_hospitalized": [covid_hospitalized_rate.mean().values / covid_infections.mean().values],
            }
        )
        covid_hospitalized_rate_summary.to_csv(
            os.path.join(output_dir, output_string, "hosptialized_rate_" + output_string + ".csv"), index=False
        )

        # plot of R0 values for validation that a variety of R0 values were selected
        plot = df[["r0_start"]].groupby("r0_start").count().reset_index().r0_start.hist()
        fig = plot.get_figure()
        fig.savefig(os.path.join(output_dir, output_string, "starting_r0_" + output_string + ".png"))

        # fill NAs with 0
        df = df.fillna(0)
        export = df

        vars_of_interest = [
            "date",
            "region",
            "r0_start",
            "newinfected",
            "cumulinfected",
            "demand_acute",
            "demand_icu",
            "demand_icu_vent",
            "hospital_census_acute",
            "hospital_census_icu",
            "all_seeking_hospital",
            "all_seeking_acute",
            "all_seeking_icu",
            "covid_seeking_hospital",
            "covseek_acute",
            "covseek_icu",
            "covseek_nh",
            "covdemand_acute",
            "covdemand_icu",
            "covdemand_icu_vent",
            "covdemand_nh",
            "acute_patients_turned_away",
            "acute_covid_patients_turned_away",
            "icu_patients_turned_away",
            "icu_covid_patients_turned_away",
            "noncovid_hospital_census_acute",
            "noncovid_hospital_census_icu",
            "covid_hospital_census_acute",
            "covid_hospital_census_icu",
        ]

        # Export the raw data
        export[vars_of_interest].to_csv(
            os.path.join(output_dir, output_string, "raw_" + output_string + ".csv"), index=False
        )

        # Aggregate data by date - by REGION
        export_ps = (
            export[["date", "region", "flag_exceeds_acute_beds", "flag_exceeds_icu_beds"]]
            .groupby(["date", "region"])
            .sum()
            .reset_index()
        )
        # Calculate probability of exceed bed counts on particular dates
        export_ps["pr_over_capacity_acute"] = export_ps["flag_exceeds_acute_beds"] / len(df.run_id.unique())
        export_ps["pr_over_capacity_icu"] = export_ps["flag_exceeds_icu_beds"] / len(df.run_id.unique())

        # Calculate mean of values
        export_beds_mean = export[vars_of_interest].groupby(["date", "region"]).mean().reset_index().round(0)
        export_beds_mean.columns = [(column + "_mean") for column in list(export_beds_mean)]

        # Calculate variance of exported values
        export_beds_var = export[vars_of_interest].groupby(["date", "region"]).var().reset_index().round()
        export_beds_var.columns = [(column + "_var") for column in list(export_beds_var)]

        # Calculate ninety percent of exported values
        export_beds_ninety = export[vars_of_interest].groupby(["date", "region"]).quantile(0.75).reset_index().round()
        export_beds_ninety.columns = [(column + "_75percentile") for column in list(export_beds_ninety)]

        # Calculate 10th percentile of exported values
        export_beds_ten = export[vars_of_interest].groupby(["date", "region"]).quantile(0.25).reset_index().round()
        export_beds_ten.columns = [(column + "_25percentile") for column in list(export_beds_ten)]

        export_regions = pd.concat(
            [
                export_ps,
                export_beds_mean.iloc[:, 1:],
                export_beds_var.iloc[:, 1:],
                export_beds_ninety.iloc[:, 1:],
                export_beds_ten.iloc[:, 1:],
            ],
            axis=1,
        )

        # Aggregate data by date for the STATE
        export_state_raw = export.groupby(["run_id", "date"]).sum().reset_index()
        export_state_raw = export_state_raw.drop(columns=["flag_exceeds_acute_beds", "flag_exceeds_icu_beds"])

        export_state_raw = export_state_raw.assign(
            flag_exceeds_acute_beds=np.where(export_state_raw["acute_patients_turned_away"] > 0, 1, 0)
        )
        export_state_raw = export_state_raw.assign(
            flag_exceeds_icu_beds=np.where(export_state_raw["icu_patients_turned_away"] > 0, 1, 0)
        )

        export_state_ps = (
            export_state_raw[["date", "flag_exceeds_acute_beds", "flag_exceeds_icu_beds"]]
            .groupby(["date"])
            .sum()
            .reset_index()
        )
        # Calculate probability of exceed bed counts on particular dates
        export_state_ps["pr_over_capacity_acute"] = export_state_ps["flag_exceeds_acute_beds"] / len(df.run_id.unique())
        export_state_ps["pr_over_capacity_icu"] = export_state_ps["flag_exceeds_icu_beds"] / len(df.run_id.unique())

        # Calculate mean of numeric values
        export_state_beds_mean = export_state_raw[vars_of_interest].groupby(["date"]).mean().reset_index().round(0)
        export_state_beds_mean.columns = [(column + "_mean") for column in list(export_state_beds_mean)]

        # Calculate variance of exported values
        export_state_beds_var = export_state_raw[vars_of_interest].groupby(["date"]).var().reset_index().round()
        export_state_beds_var.columns = [(column + "_var") for column in list(export_state_beds_var)]

        # Calculate ninety percent of exported values
        export_state_beds_ninety = (
            export_state_raw[vars_of_interest].groupby(["date"]).quantile(0.75).reset_index().round()
        )
        export_state_beds_ninety.columns = [(column + "_75percentile") for column in list(export_state_beds_ninety)]

        # Calculate 10th percentile of exported values
        export_state_beds_ten = (
            export_state_raw[vars_of_interest].groupby(["date"]).quantile(0.25).reset_index().round()
        )
        export_state_beds_ten.columns = [(column + "_25percentile") for column in list(export_state_beds_ten)]

        export_state = pd.concat(
            [
                export_state_ps,
                export_state_beds_mean.iloc[:, 1:],
                export_state_beds_var.iloc[:, 1:],
                export_state_beds_ninety.iloc[:, 1:],
                export_state_beds_ten.iloc[:, 1:],
            ],
            axis=1,
        )
        export_state["region"] = "State"

        # concatenate regional and state estimates
        export_df = pd.concat([export_regions, export_state], sort=True)

        # export aggregated CSV
        export_df["forecast_day"] = export_df.date

        export_df["scenario"] = output_name
        export_df["numruns"] = len(df.run_id.unique())
        # condiiton was requested, but has not yet been used
        Re_min = output_name.split("_")[1]
        Re_max = output_name.split("_")[2]
        export_df["Re_range"] = str(Re_min).replace("p", ".") + " - " + str(Re_max).replace("p", ".")
        export_df["case_mult"] = output_name.split("_")[3]
        export_df.to_csv(
            os.path.join(output_dir, output_string, "orig_aggregated_" + output_string + ".csv"), index=False
        )

        # specify order of variables and variables output
        export_df[
            [
                "forecast_day",
                "region",
                "numruns",
                "scenario",
                "Re_range",
                "case_mult",
                "newinfected_mean",
                "cumulinfected_mean",
                "demand_acute_mean",
                "demand_acute_25percentile",
                "demand_acute_75percentile",
                "demand_icu_mean",
                "demand_icu_25percentile",
                "demand_icu_75percentile",
                "demand_icu_vent_mean",
                "demand_icu_vent_25percentile",
                "demand_icu_vent_75percentile",
                "hospital_census_acute_mean",
                "hospital_census_icu_mean",
                "hospital_census_acute_25percentile",
                "hospital_census_icu_25percentile",
                "hospital_census_acute_75percentile",
                "hospital_census_icu_75percentile",
                "covdemand_acute_mean",
                "covdemand_icu_mean",
                "covdemand_icu_vent_mean",
                "covdemand_nh_mean",
                "newinfected_75percentile",
                "cumulinfected_75percentile",
                "newinfected_25percentile",
                "cumulinfected_25percentile",
                "covdemand_acute_75percentile",
                "covdemand_acute_25percentile",
                "covdemand_icu_75percentile",
                "covdemand_icu_25percentile",
                "covdemand_icu_vent_75percentile",
                "covdemand_icu_vent_25percentile",
                "covdemand_nh_75percentile",
                "covdemand_nh_25percentile",
                "covseek_acute_mean",
                "covseek_icu_mean",
                "covseek_acute_75percentile",
                "covseek_icu_75percentile",
                "covseek_acute_25percentile",
                "covseek_icu_25percentile",
                "covid_hospital_census_acute_mean",
                "covid_hospital_census_acute_75percentile",
                "covid_hospital_census_acute_25percentile",
                "covid_hospital_census_icu_mean",
                "covid_hospital_census_icu_75percentile",
                "covid_hospital_census_icu_25percentile",
                "noncovid_hospital_census_acute_mean",
                "noncovid_hospital_census_acute_75percentile",
                "noncovid_hospital_census_acute_25percentile",
                "noncovid_hospital_census_icu_mean",
                "noncovid_hospital_census_icu_75percentile",
                "noncovid_hospital_census_icu_25percentile",
            ]
        ].to_csv(os.path.join(output_dir, output_string, "aggregated_" + output_string + ".csv"), index=False)

    # From aggregate all results:
    master_df = pd.DataFrame()
    master_df_hosp_rate = pd.DataFrame()

    dirlist = os.listdir(output_dir)
    for directory in dirlist:

        # aggregate results files
        for file in glob.glob(os.path.join(output_dir, directory, "aggregated_*.csv")):
            temp_df = pd.read_csv(file)
            master_df = pd.concat([master_df, temp_df])

        # aggregate hospitalization rates
        for file in glob.glob(os.path.join(output_dir, directory, "hosptialized_rate_*.csv")):
            temp_df_hosp_rate = pd.read_csv(file)
            master_df_hosp_rate = pd.concat([master_df_hosp_rate, temp_df_hosp_rate])

    master_df.to_csv(os.path.join(output_dir, "master_results.csv"), index=False)
    master_df_hosp_rate.to_csv(os.path.join(output_dir, "master_hosp_rate.csv"), index=False)

    # Calculate smooth version of df with 7 day window
    master_df["forecast_day"] = pd.to_datetime(master_df["forecast_day"], format="%Y-%m-%d")
    df_to_smooth = master_df.sort_values(by=["scenario", "region", "forecast_day"])

    smoothed_df = pd.DataFrame()
    for scenario in master_df["scenario"].unique():
        for region in master_df["region"].unique():
            temp_df = df_to_smooth[(df_to_smooth.scenario == scenario) & (df_to_smooth.region == region)]
            temp_df_numeric = temp_df.select_dtypes(include=np.number)
            temp_df_numeric_smoothed = temp_df_numeric.apply(lambda x: x.rolling(7, min_periods=1).mean())
            temp_df_numeric_smoothed_int = temp_df_numeric_smoothed.astype(int)
            temp_df = pd.concat([temp_df.select_dtypes(exclude=np.number), temp_df_numeric_smoothed_int], axis=1)
            smoothed_df = pd.concat([smoothed_df, temp_df], axis=0)

    smoothed_df.to_csv(os.path.join(output_dir, "smoothed_master_results.csv"), index=False)
