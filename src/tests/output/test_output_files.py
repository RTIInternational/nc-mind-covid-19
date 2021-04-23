from src.tests.fixtures import multiple_run_results_regular, region_list
import glob
import pandas as pd
from datetime import datetime
import pytest
from datetime import datetime as dt


@pytest.mark.slow
def test_raw_output_files(multiple_run_results_regular, region_list):
    """This tests the raw output for each run."""
    output_folder = multiple_run_results_regular["output_dir"]
    print(output_folder)

    files = glob.glob(f"{output_folder}/output */raw_output *.csv")
    print(files)
    dfs = []
    for i in range(len(files)):
        tmp = pd.read_csv(files[i])
        tmp["run_id"] = i
        dfs.append(tmp)

    df = pd.concat(dfs)

    # check all the regions appear
    for run_id, run_group in df.groupby("run_id"):
        regions = run_group["region"].astype(str).unique()
        assert set(region_list["regions"]) == set(regions)

    for index, region_group in df.groupby(["run_id", "region"]):
        # check we have the correct number of days. Note we leave out day 0
        days = region_group["date"].unique()
        assert len(days) == multiple_run_results_regular["days"] - 1

        datetime_days = region_group["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d")).tolist()

        cuminf = region_group["cumulinfected"].tolist()
        newinf = region_group["newinfected"].tolist()
        for i in range(1, len(cuminf)):
            # check that the dates are in increasing order
            assert datetime_days[i] > datetime_days[i - 1]
            # test that cumulinfected[i] ~= newinfected[i] + cumulinfected[i-1]
            assert abs(cuminf[i] - (cuminf[i - 1] + newinf[i])) <= 1

    # test that all of the values are non-negative
    for col in df:
        if col in ["date", "run_id", "region"]:
            continue
        assert all(df[col].astype(int) >= 0)

    # check the covid counts <= general counts
    general_covid_pairs = [
        ["icu_patients_turned_away", "icu_covid_patients_turned_away"],
        ["acute_patients_turned_away", "acute_covid_patients_turned_away"],
        ["all_seeking_hospital", "covid_seeking_hospital"],
        ["hospital_census_acute", "covid_hospital_census_acute"],
        ["hospital_census_icu", "covid_hospital_census_icu"],
    ]
    for col_pair in general_covid_pairs:
        diff = df[col_pair[0]] - df[col_pair[1]]
        assert all(diff >= 0)


@pytest.mark.slow
def test_aggregated_run_output_files(multiple_run_results_regular, region_list):
    """This tests the aggregated output from the first level of
    aggregation (across runs within an iteration)
    """
    output_folder = multiple_run_results_regular["output_dir"]

    files = glob.glob(f"{output_folder}/output */aggregated_output *.csv")
    df = pd.concat([pd.read_csv(f) for f in files])

    # check all the regions appear
    for run_id, run_group in df.groupby("scenario"):
        regions = run_group["region"].astype(str).unique()
        assert "State" in regions
        assert set(region_list["regions"]) == set(regions) - set(["State"])

    for index, region_group in df.groupby(["scenario", "region"]):
        # check we have the correct number of days. Note we leave out day 0
        days = region_group["forecast_day"].unique()
        assert len(days) == multiple_run_results_regular["days"] - 1

        datetime_days = region_group["forecast_day"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d")).tolist()

        for i in range(1, len(datetime_days)):
            # check that the dates are in increasing order
            assert datetime_days[i] > datetime_days[i - 1]

    # test that all of the values are non-negative
    for col in df:
        if col in [
            "forecast_day",
            "scenario",
            "region",
            "condition",
            "Re_range",
            "case_mult",
        ]:
            continue
        assert all(df[col].astype(int) >= 0)

    # check the percentiles 10 <= mean <= 90
    mean_cols = [
        "covdemand_acute",
        "covdemand_icu",
        "covdemand_nh",
        "demand_acute",
        "demand_icu",
        "hospital_census_acute",
        "hospital_census_icu",
        "covid_hospital_census_acute",
        "covid_hospital_census_icu",
        "cumulinfected",
        "newinfected",
        "noncovid_hospital_census_acute",
        "noncovid_hospital_census_icu",
    ]
    for col in mean_cols:
        assert all(df[f"{col}_25percentile"] <= df[f"{col}_mean"])
        assert all(df[f"{col}_mean"] <= df[f"{col}_75percentile"])


@pytest.mark.slow
def test_master_results_output_files(multiple_run_results_regular, region_list):
    """This tests the aggregated output from the highest level of aggregation
    (aggregating the aggregated iterations)
    """
    output_folder = multiple_run_results_regular["output_dir"]

    df_unsmoothed = pd.read_csv(f"{output_folder}/master_results.csv")
    df_smoothed = pd.read_csv(f"{output_folder}/smoothed_master_results.csv")

    for df in [df_unsmoothed, df_smoothed]:

        # check all the regions appear
        for run_id, run_group in df.groupby("scenario"):
            regions = run_group["region"].astype(str).unique()
            assert "State" in regions
            assert set(region_list["regions"]) == set(regions) - set(["State"])

        for index, region_group in df.groupby(["scenario", "region"]):
            # check we have the correct number of days. Note we leave out day 0
            days = region_group["forecast_day"].unique()
            assert len(days) == multiple_run_results_regular["days"] - 1

            datetime_days = region_group["forecast_day"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d")).tolist()

            for i in range(1, len(datetime_days)):
                # check that the dates are in increasing order
                assert datetime_days[i] > datetime_days[i - 1]

        # test that all of the values are non-negative
        for col in df:
            if col in [
                "forecast_day",
                "scenario",
                "region",
                "condition",
                "Re_range",
                "case_mult",
            ]:
                continue
            assert all(df[col].astype(int) >= 0)

        # check the percentiles 10 <= mean <= 90
        mean_cols = [
            "covdemand_acute",
            "covdemand_icu",
            "covdemand_nh",
            "demand_icu",
            "demand_acute",
            "hospital_census_acute",
            "hospital_census_icu",
            "covid_hospital_census_acute",
            "covid_hospital_census_icu",
            "cumulinfected",
            "newinfected",
            "noncovid_hospital_census_acute",
            "noncovid_hospital_census_icu",
        ]
        for col in mean_cols:
            assert all(df[f"{col}_25percentile"] <= df[f"{col}_mean"])
            assert all(df[f"{col}_mean"] <= df[f"{col}_75percentile"])

        # check that the percent cols are between 0 and 1
        if pd.Series(["pr_over_capacity_acute", "pr_over_capacity_icu"]).isin(df.columns).all():

            for perc_col in ["pr_over_capacity_acute", "pr_over_capacity_icu"]:
                assert all(df[perc_col] >= 0)
                assert all(df[perc_col] <= 1)
