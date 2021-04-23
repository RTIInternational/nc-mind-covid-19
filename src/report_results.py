import numpy as np
import pandas as pd

from src.constants import POPULATION
from src.calibration_collection import format_beds_taken
from src.state import COVIDState


def filter_beds_taken(beds_taken, icu_status: int = None, covid_statuses: list = None, region: int = None):
    if isinstance(region, int):
        beds_taken = beds_taken[beds_taken["Region"] == region]
    if isinstance(icu_status, int):
        beds_taken = beds_taken[beds_taken["ICU_STATUS"] == icu_status]
    if isinstance(covid_statuses, list):
        beds_taken = beds_taken[beds_taken["COVID_STATUS"].isin(covid_statuses)]
    return beds_taken.drop(["LOCATION", "ICU_STATUS", "COVID_STATUS", "Region"], axis=1).sum()


class ReportResults:
    """
    A class for creating the CSVs requested by the Governor
    """

    def __init__(self, ldm):

        # Base Information
        self.ldm = ldm
        self.r0 = ldm.params["covid19"]["r0"]
        self.h_ids = self.ldm.population.Start_Location.value_counts()
        self.ids = [an_id for an_id in self.h_ids.index if an_id in self.ldm.nodes.all_hospitals]
        self.multiplier = POPULATION / self.ldm.population.shape[0]
        self.time_start = 0
        self.days_to_run = ldm.params["base"]["time_horizon"]
        self.total_days = self.time_start + self.days_to_run
        self.run_id = str(ldm.run_dir).split("/")[-1]

    def results_singular_run(self):
        """
        Summarize results from a singular run in a tabular format to be exported
        as a CSV.
        """

        movement = self.ldm.movement.location.events.make_events()
        movement["Region"] = [self.ldm.regions[int(unique_id)] for unique_id in movement.Unique_ID]
        movement["To_Hospital"] = movement["New Location"].isin(self.ldm.movement.all_hospitals)
        movement["ICU"] = movement["ICU"].astype(bool)

        cases = self.ldm.disease.covid_cases.make_events()
        cases["Region"] = [self.ldm.regions[int(unique_id)] for unique_id in cases.Unique_ID]
        cases["ICU"] = cases["ICU"].astype(bool)

        bt = format_beds_taken(self.ldm)
        bt = bt.reset_index()
        hospital_bt = bt[bt["LOCATION"].isin(self.ldm.movement.all_hospitals)].copy()
        hospital_bt["Region"] = [
            self.ldm.unique_regions[self.ldm.movement.facilities[i].region_str] for i in hospital_bt["LOCATION"]
        ]

        covid_to_nh_dict = self.ldm.disease.nh_demand_by_day

        pcta = self.ldm.movement.patients_completely_turned_away.make_events()
        pcta["ICU"] = pcta["ICU"].astype(bool)
        pcta["Region"] = [self.ldm.regions[int(unique_id)] for unique_id in pcta.Unique_ID]
        df = pd.DataFrame()

        ventilator_covid_p = self.ldm.disease.params["icu_with_ventilator_p"]
        ventilator_noncovid_p = self.ldm.movement.params["ventilator_probability"]

        # create temporary results dataframe by region
        for region in self.ldm.disease.covid_beds_by_day.keys():

            temp_df = pd.DataFrame(index=range(0, self.total_days))
            # all patients (COVID and non-COVID)
            temp_df["all_seeking_hospital"] = (
                movement[(movement.To_Hospital) & (movement.Region == region)].groupby(by="Time").size()
            )
            temp_df["all_seeking_acute"] = (
                movement[(movement.To_Hospital) & (movement.Region == region) & (~movement.ICU)]
                .groupby(by="Time")
                .size()
            )
            temp_df["all_seeking_icu"] = (
                movement[(movement.To_Hospital) & (movement.Region == region) & (movement.ICU)]
                .groupby(by="Time")
                .size()
            )
            # new COVID infections (not just reported cases)
            temp_df["newinfected"] = cases[cases.Region == region].groupby("Time").size()
            # COVID infections seeking hospital beds
            temp_df["covid_seeking_hospital"] = (
                cases[(cases.Region == region) & (cases["Seeking Hospital"] == 1)].groupby(by="Time").size()
            )
            temp_df["covseek_acute"] = (
                cases[(cases.Region == region) & (cases["Seeking Hospital"] == 1) & (~cases.ICU)]
                .groupby(by="Time")
                .size()
            )
            temp_df["covseek_icu"] = (
                cases[(cases.Region == region) & (cases["Seeking Hospital"] == 1) & (cases.ICU)]
                .groupby(by="Time")
                .size()
            )

            temp_df["covseek_nh"] = pd.Series(covid_to_nh_dict[region]["daily_patients"]).astype(int)

            temp_df["covid_hospital_census_acute"] = filter_beds_taken(
                hospital_bt,
                region=region,
                icu_status=0,
                covid_statuses=[COVIDState.COVID19MILD, COVIDState.COVID19SEVERE, COVIDState.COVID19CRITICAL],
            )

            temp_df["covid_hospital_census_icu"] = filter_beds_taken(
                hospital_bt,
                region=region,
                icu_status=1,
                covid_statuses=[COVIDState.COVID19MILD, COVIDState.COVID19SEVERE, COVIDState.COVID19CRITICAL],
            )

            temp_df["hospital_census_acute"] = filter_beds_taken(hospital_bt, region=region, icu_status=0)
            temp_df["hospital_census_icu"] = filter_beds_taken(hospital_bt, region=region, icu_status=1)

            temp_df["noncovid_hospital_census_acute"] = (
                temp_df["hospital_census_acute"] - temp_df["covid_hospital_census_acute"]
            )
            temp_df["noncovid_hospital_census_icu"] = (
                temp_df["hospital_census_icu"] - temp_df["covid_hospital_census_icu"]
            )

            # map region names to region values
            region_map = list(self.ldm.unique_regions.keys())[list(self.ldm.unique_regions.values()).index(region)]
            temp_df["region"] = region_map
            temp_pta = pcta[pcta.Region == region]

            # Calculate the number of patients turned away, by bed type
            temp_df["acute_patients_turned_away"] = temp_pta[~temp_pta.ICU].groupby(by=["Time"]).size()
            temp_df["acute_covid_patients_turned_away"] = (
                temp_pta[(~temp_pta.ICU) & temp_pta.COVID_Type].groupby(by=["Time"]).size()
            )
            temp_df["icu_patients_turned_away"] = temp_pta[temp_pta.ICU].groupby(by=["Time"]).size()
            temp_df["icu_covid_patients_turned_away"] = (
                temp_pta[temp_pta.ICU & temp_pta.COVID_Type].groupby(by=["Time"]).size()
            )

            temp_df = temp_df.fillna(0)
            temp_df["cumulinfected"] = temp_df["newinfected"].cumsum()
            temp_df["covdemand_nh"] = temp_df["covseek_nh"].cumsum()
            temp_df["demand_acute"] = temp_df["hospital_census_acute"] + temp_df["acute_patients_turned_away"]
            temp_df["demand_icu"] = temp_df["hospital_census_icu"] + temp_df["icu_patients_turned_away"]
            temp_df["covdemand_acute"] = (
                temp_df["acute_covid_patients_turned_away"] + temp_df["covid_hospital_census_acute"]
            )
            temp_df["covdemand_icu"] = temp_df["icu_covid_patients_turned_away"] + temp_df["covid_hospital_census_icu"]
            for i in range(1, 5):
                temp_df["demand_acute"] += round(temp_df["acute_patients_turned_away"].shift(i).fillna(0) * (1 - i / 5))
                temp_df["demand_icu"] += round(temp_df["icu_patients_turned_away"].shift(i).fillna(0) * (1 - i / 5))
                temp_df["covdemand_acute"] += round(
                    temp_df["acute_covid_patients_turned_away"].shift(i).fillna(0) * (1 - i / 5)
                )
                temp_df["covdemand_icu"] += round(
                    temp_df["icu_covid_patients_turned_away"].shift(i).fillna(0) * (1 - i / 5)
                )

            temp_df["covdemand_icu_vent"] = round(temp_df["covdemand_icu"] * ventilator_covid_p)
            temp_df["demand_icu_vent"] = (
                round((temp_df["demand_icu"] - temp_df["covdemand_icu"]) * ventilator_noncovid_p)
                + temp_df["covdemand_icu_vent"]
            )

            df = pd.concat([df, temp_df])

        # apply the multiplier to the dataframe and round to integers
        df.loc[:, df.columns != "region"] = df.loc[:, df.columns != "region"].mul(self.multiplier)
        df = df.round(0)
        # get day column
        df = df.reset_index(drop=False)
        df.rename(columns={df.columns[0]: "day"}, inplace=True)
        df.day = df.day - self.time_start
        # add run parameters
        df["r0_start"] = self.r0
        df["run_id"] = self.run_id

        # flag if any agents are turned away
        df = df.assign(flag_exceeds_acute_beds=np.where(df["acute_patients_turned_away"] > 0, 1, 0))
        df = df.assign(flag_exceeds_icu_beds=np.where(df["icu_patients_turned_away"] > 0, 1, 0))

        # order columns in a logical manner
        df = df[
            [
                "day",
                "run_id",
                "r0_start",
                "region",
                "newinfected",
                "cumulinfected",
                "demand_acute",
                "demand_icu",
                "demand_icu_vent",
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
                # "covid_pta_needs_acute",
                "icu_patients_turned_away",
                "icu_covid_patients_turned_away",
                # "covid_pta_needs_icu",
                "flag_exceeds_acute_beds",
                "flag_exceeds_icu_beds",
                "hospital_census_acute",
                "hospital_census_icu",
                "noncovid_hospital_census_acute",
                "noncovid_hospital_census_icu",
                "covid_hospital_census_acute",
                "covid_hospital_census_icu",
            ]
        ]

        # df = df[df.day >= 0]

        return df
