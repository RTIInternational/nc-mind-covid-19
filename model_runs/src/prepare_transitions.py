import argparse as arg
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import src.data_input as di
from src.parameters import load_parameters

FILEPATH_YAML = "model_runs/config/filepaths.yaml"
filepaths = yaml.load(Path(FILEPATH_YAML).read_text(), Loader=yaml.loader.SafeLoader)

parameters = {
    "LOS_mean": 5,  # ------------ Source: Parameters file, see LARGE or SMALL. Average LOS across all hospitals was 5.
    "community_to_nh": 100_000,  # Source: Phone call with CDC, we were told 93k and rounded to 100k.
    "nh_death": 0.15,  # --------- Source:  TODO
    "nh_to_com": 0.673,  # ------- CDC email: 67.3% of NH patients return to the community, remaining % go to STACH
    "unc_to_unc": 0.90,  # ------- Best guess from Epi Team - updated after phone call with UNC collaborators
    "unc_to_large": 0.08,  # ----- 80% of remaining 10% go to large, see "large_to_large" parameter below
    "unc_to_small": 0.02,  # ----- 20% of remaining 10% go to small, see "large_to_small" parameter below
    "nonunc_to_unc": 0.02,  # ---- Best guess from RTI. Almost noone goes nonunc to unc
    "large_to_large": 0.80,  # --- Best guess from Epi Team - updated after phone call with UNC collaborators
    "large_to_small": 0.20,  # --- Best guess from Epi Team - updated after phone call with UNC collaborators
    "small_to_large": 0.90,  # --- Best guess from Epi Team - updated after phone call with UNC collaborators
    "small_to_small": 0.10,  # --- Best guess from Epi Team - updated after phone call with UNC collaborators
    "lt_to_stach": 0.071,  # ----- Toth_2017_CID_Potential Interventions LTACH Reduce Transmission CRE.pdf
    "lt_to_nh": 0.449,  # -------- Toth_2017_CID_Potential Interventions LTACH Reduce Transmission CRE.pdf
    "lt_death": 0.01,  # --------- Toth_2017_CID_Potential Interventions LTACH Reduce Transmission CRE.pdf
    "lt_65p": 0.75,  # ----------- TODO: Estimate from RTI. We need lots of LT to be 65p for NH reasons.
    "nh_st_nh": 0.80,  # --------- 80% of NH to STACH movement we return to a NH
    "n_beds_ltachs": 479,  # ----- Source: TODO
    "n_beds_nhs": 49_318,  # ----- Source: TODO
    "ltach_fill_p": 0.9,  # ------ Source: TODO
    "nh_fill_p": 0.7,  # --------- Source: TODO
    "readmission": 0.094,  # ------ 9.4% of people readmitted to UNC within 30 days
}


def add_total(df):
    df["Total"] = df.sum(axis=1)
    return df


def distribute_discharges(discharges: pd.DataFrame, a0_p: float, a1_p: float, a2_p: float) -> list:
    """Given the number of admissions required for a specific hospital, find the number of people from each age group
    that need to go to that hospital from each county.

    Args:
        discharges (pd.DataFrame): A discharge DataFrame from the SHEPS center
        a0_p (float): The overall probability that someone in that facility type is 0 < 50 years old
        a1_p (float): The overall probability that someone in that facility type is 50 < 65 years old
        a2_p (float): The overall probability that someone in that facility type is 65+ years old
    """
    a_list = []
    for county in counties:
        value = discharges.loc[county, "From Comm - Adjusted"]
        a0 = value * a0_p
        a1 = value * a1_p
        a2 = value * a2_p
        a_list.extend([a0, a1, a2])
    return a_list


def stach_transitions(
    hospital: str, breakdown: pd.DataFrame, category: str, p_unc: float, p_large: float, p_small: float
) -> list:
    """Calulate the stach discharge probabilities for a specific hospital.

    Args:
        hospital (str): The name of the hospital (as it appears in the SHEPS data)
        df_breakdown (pd.DataFrame): A demographic breakdown of discharges from a hospital - provided by SHEPS
        category (str): The facility type category. One of: UNC, LARGE, or SMALL
        p_unc (float): What percent of counties that have CATEGORY patients, also have UNC patients
        p_large (float): What percent of counties that have CATEGORY patients, also have LARGE patients
        p_small (float): What percent of counties that have CATEGORY patients, also have SMALL patients
    """
    x_to_unc = sbs.loc[category, "UNC"]
    x_to_large = sbs.loc[category, "LARGE"]
    x_to_small = sbs.loc[category, "SMALL"]
    x_to_nh = sbs.loc[category, "NH"] - sbs.loc["NH", category] * parameters["nh_st_nh"]
    x_to_lt = sbs.loc[category, "LT"]
    x_movement = sbs.loc[category].sum() - sbs.loc["NH", category] * parameters["nh_st_nh"]

    temp_total = breakdown[hospital].max()
    a1 = breakdown.loc[age_1_columns, hospital].sum() * 0.75 / temp_total
    a2 = breakdown.loc[age_2_columns, hospital].sum() / temp_total

    rows = []
    for county_code in counties:
        for age_group in age_groups:
            if category == "UNC":
                to_unc = x_to_unc / x_movement if county_code in unc_counties else 0
                to_large = (x_to_large / x_movement) * (1 / p_large) if county_code in large_counties else 0
                to_small = ((x_to_unc + x_to_large + x_to_small) / x_movement) - to_unc - to_large
            elif category == "LARGE":
                to_unc = (x_to_unc / x_movement) * (1 / p_unc) if county_code in unc_counties else 0
                to_large = (x_to_large / x_movement) if county_code in large_counties else 0
                to_small = ((x_to_unc + x_to_large + x_to_small) / x_movement) - to_unc - to_large
            elif category == "SMALL":
                to_unc = (x_to_unc / x_movement) * (1 / p_unc) if county_code in unc_counties else 0
                to_large = (x_to_large / x_movement) * (1 / p_large) if county_code in large_counties else 0
                to_small = ((x_to_unc + x_to_large + x_to_small) / x_movement) - to_unc - to_large
            else:
                raise ValueError(f"Category {category} does not belong.")
            if county_code not in small_counties:
                to_small = 0
                to_large = ((x_to_unc + x_to_large + x_to_small) / x_movement) - to_unc
            if age_group == 2:
                # LT patients are primarily 65p: we also need more 65p so we can have enough LT to NH movement
                to_lt = (parameters["lt_65p"] / a2) * (x_to_lt / x_movement)
                # All NH movement must be 65+
                to_nh = (1 / a2) * (x_to_nh / x_movement)
            elif age_group == 1:
                to_lt = ((1 - parameters["lt_65p"]) / a1) * (x_to_lt / x_movement)
                to_nh = 0
            else:
                to_nh, to_lt = 0, 0
            row = [0, to_unc, to_large, to_small, to_lt, to_nh]
            row[0] = 1 - sum(row)
            row = [county_code, age_group, hospital] + row
            rows.append(row)
    return rows


def find_los(items, params):
    los = []
    for item in items:
        if item in params["location"]["facilities"].keys():
            los.append(params["location"]["facilities"][item]["los"]["mean"])
        else:
            los.append(params["location"]["facilities"]["LARGE"]["los"]["mean"])
    return los


# ----------------------------------------------------------------------------------------------------------------------
# Preperation ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    description = """Create transition probabilities using ."""
    parser = arg.ArgumentParser(description=description)
    parser.add_argument("--data_file", help="The filepath to additional data", default=None)

    args = parser.parse_args()
    counties = list(range(1, 201, 2))
    age_groups = [0, 1, 2]

    hospital_info = di.hospital_info()
    hospital_beds = di.hospital_beds().set_index("Name")

    # Synthetic Population ---------------------------------------------------------------------------------------------
    syn_pop = di.read_population()
    syn_pop_counts = syn_pop.groupby(["County_Code", "Age_Group"]).size()
    syn_pop_counts.columns = ["Population"]

    # File Paths
    location_transitions_path = filepaths["location_transitions"]["path"]
    community_transitions_path = filepaths["community_transitions"]["path"]

    # If additional data provided --------------------------------------------------------------------------------------
    if args.data_file:
        print("Additional data provided. Thank you.")
        df = pd.read_csv(args.data_file)
        # Only keep known hospitals
        crosswalk = di.hospital_crosswalk()
        hospital_df = df[df["MedSurge Name"].isin(crosswalk["MedSurge Name"].unique())].copy()
        name_map = {}
        for _, row in crosswalk.iterrows():
            name_map[row["MedSurge Name"]] = row.Name
        name_map["Johnston UNC Health Care"] = "Johnston Health Clayton"

        hospital_df["Name"] = hospital_df.hospital_name.map(name_map)

        hospital_df = hospital_df.drop(["hospital_name", "MedSurge Name"], axis=1).set_index("Name")
        hospital_df = hospital_df.merge(
            hospital_beds[["Acute Beds", "ICU Beds", "Category"]], left_index=True, right_index=True
        )
        hospital_df = hospital_df.rename(
            columns={"non_covid_acute_count": "Acute Agents", "non_covid_icu_count": "ICU Agents"}
        )
    # If no additional data provided -----------------------------------------------------------------------------------
    else:
        print("\n\n\nWARNING: No additional data was provided.")
        print("Creating default transition probabilities.\n\n\n")

        hospital_df = hospital_beds.copy()
        hospital_df = hospital_df[["Category", "Total Beds", "Acute Beds", "ICU Beds"]].copy()
        hospital_df["Acute Agents"] = hospital_df["Acute Beds"] * 0.65
        hospital_df["ICU Agents"] = hospital_df["ICU Beds"] * 0.5
        hospital_df["All COVID-19 Non-ICU Patients in Hospital"] = 0
        hospital_df["Adult ICU Covid-19 Positive Patients"] = 0

    # ----- Set BEDS = PATIENTS if too many patients are reported
    test_df = hospital_df.copy()
    total_icu = hospital_df["ICU Agents"] + hospital_df["Adult ICU Covid-19 Positive Patients"]
    temp = hospital_df[total_icu > hospital_df["ICU Beds"]].copy()
    count_switched = total_icu - temp["ICU Beds"]
    # --- Fix ICU Beds
    temp["ICU Beds"] = total_icu
    # --- Lower Acute Beds to match the ICU beds added
    temp["Acute Beds"] -= count_switched
    # --- Update main DataFrame
    hospital_df.loc[temp.index] = temp

    # --- Test that we didn't mess up:
    assert hospital_df[["ICU Beds", "Acute Beds"]].sum().sum() == test_df[["ICU Beds", "Acute Beds"]].sum().sum()
    assert sum(hospital_df["ICU Agents"] > hospital_df["ICU Beds"]) == 0

    # --- OK: Continue
    params = load_parameters(Path("model_runs/location/run_calibration/parameters.json"))
    hospital_df["Avg_LOS"] = find_los(hospital_df.index, params)
    hospital_df["Yearly"] = hospital_df[["Acute Agents", "ICU Agents"]].sum(axis=1).div(hospital_df["Avg_LOS"]) * 365
    # SHEPS row names --------------------------------------------------------------------------------------------------
    community_columns = [
        "Patient Disposition Home, self, or outpatient care",
        "Patient Disposition Discharged, transferred to psychiatric facility",
        "Patient Disposition Hospice",
        "Patient Disposition Left against medical advice",
        "Patient Disposition Court/Law Enforcement",
        "Patient Disposition Other/Unknown",
    ]
    stach_columns = [
        "Patient Disposition Discharged, transferred to acute facility",
        "Patient Disposition Discharged, transferred",
    ]
    lt_columns = ["Patient Disposition Discharged, transferred to long term acute care facility (LTAC)"]
    nh_columns = [
        "Patient Disposition Discharged, transferred to facility that provides nursing, custodial, or supportive care"
    ]
    death_columns = ["Patient Disposition Expired"]
    age_0_columns = ["Age Group Less than 1 Year", "Age Group 1 - 17 years", "Age Group 18 - 44 years"]
    age_1_columns = ["Age Group 45 - 64 years"]
    age_2_columns = ["Age Group 65 - 84 years", "Age Group 85 or more years"]

    # SHEPS Discharges -------------------------------------------------------------------------------------------------
    # By County
    discharges = di.county_discharges()
    unc_discharges = add_total(discharges[list(hospital_beds[hospital_beds.Category == "UNC"].index)].copy())
    large_discharges = add_total(discharges[list(hospital_beds[hospital_beds.Category == "LARGE"].index)].copy())
    small_discharges = add_total(discharges[list(hospital_beds[hospital_beds.Category == "SMALL"].index)].copy())
    # By Demographic
    breakdown = di.demographics().set_index("Category")
    unc_breakdown = add_total(breakdown[list(hospital_beds[hospital_beds.Category == "UNC"].index)].copy())
    large_breakdown = add_total(breakdown[list(hospital_beds[hospital_beds.Category == "LARGE"].index)].copy())
    small_breakdown = add_total(breakdown[list(hospital_beds[hospital_beds.Category == "SMALL"].index)].copy())

    # UNC Discharges
    unc_counties = list(unc_discharges[unc_discharges.Total > 0].index)
    unc_total = unc_breakdown.loc[["Patient Residence State NC", "Patient Residence State Not NC"]]["Total"].sum()
    unc_50_64 = unc_breakdown.loc[age_1_columns, "Total"].sum() * 0.75 / unc_total
    unc_G65 = unc_breakdown.loc[age_2_columns, "Total"].sum() / unc_total
    unc_L50 = 1 - unc_50_64 - unc_G65
    # LARGE Discharges
    large_counties = list(large_discharges[large_discharges.Total > 0].index)
    large_total = large_breakdown.loc[["Patient Residence State NC", "Patient Residence State Not NC"]]["Total"].sum()
    large_50_64 = large_breakdown.loc[age_1_columns, "Total"].sum() * 0.75 / large_total
    large_G65 = large_breakdown.loc[age_2_columns, "Total"].sum() / large_total
    large_L50 = 1 - large_50_64 - large_G65
    # SMALL Discharges
    small_counties = list(small_discharges[small_discharges.Total > 0].index)
    small_total = small_breakdown.loc[["Patient Residence State NC", "Patient Residence State Not NC"]]["Total"].sum()
    small_50_64 = small_breakdown.loc[age_1_columns, "Total"].sum() * 0.75 / small_total
    small_G65 = small_breakdown.loc[age_2_columns, "Total"].sum() / small_total
    small_L50 = 1 - small_50_64 - small_G65

    # Population of each category catchment area
    pop = syn_pop_counts.reset_index()
    pop.columns = ["County_Code", "Age_Group", "Population"]
    pop = pop.groupby("County_Code")["Population"].sum()
    pop = pd.DataFrame(pop).merge(unc_discharges["Total"], left_index=True, right_index=True)
    pop = pd.DataFrame(pop).merge(large_discharges["Total"], left_index=True, right_index=True)
    pop = pd.DataFrame(pop).merge(small_discharges["Total"], left_index=True, right_index=True)
    pop.columns = ["Population", "UNC_Discharges", "LARGE_Discharges", "SMALL_Discharges"]
    unc_population = pop.loc[unc_counties].Population.sum()
    large_population = pop.loc[large_counties].Population.sum()
    small_population = pop.loc[small_counties].Population.sum()
    total_population = pop.Population.sum()

    # ------------------------------------------------------------------------------------------------------------------
    # THE SIX BY SIX ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # UNC --------------------------------------------------------------------------------------------------------------
    unc_community_proportion = unc_breakdown.loc[community_columns, "Total"].sum() / unc_total
    unc_stach_proportion = unc_breakdown.loc[stach_columns, "Total"].sum() / unc_total
    unc_lt_proportion = unc_breakdown.loc[lt_columns, "Total"].sum() / unc_total
    unc_nh_proportion = unc_breakdown.loc[nh_columns, "Total"].sum() / unc_total
    unc_dead_proportion = unc_breakdown.loc[death_columns, "Total"].sum() / unc_total
    # -----
    unc_admissions = hospital_df[hospital_df.Category == "UNC"].Yearly.sum()
    unc_to_com = unc_admissions * unc_community_proportion
    unc_to_stach = unc_admissions * unc_stach_proportion
    unc_to_unc = unc_to_stach * parameters["unc_to_unc"]
    unc_to_large = unc_to_stach * parameters["unc_to_large"]
    unc_to_small = unc_to_stach * parameters["unc_to_small"]
    unc_to_ltach = unc_admissions * unc_lt_proportion
    unc_to_nh = unc_admissions * unc_nh_proportion
    unc_to_death = unc_admissions * unc_dead_proportion
    # LARGE ------------------------------------------------------------------------------------------------------------
    large_community_proportion = large_breakdown.loc[community_columns, "Total"].sum() / large_total
    large_stach_proportion = large_breakdown.loc[stach_columns, "Total"].sum() / large_total
    large_lt_proportion = large_breakdown.loc[lt_columns, "Total"].sum() / large_total
    large_nh_proportion = large_breakdown.loc[nh_columns, "Total"].sum() / large_total
    large_dead_proportion = large_breakdown.loc[death_columns, "Total"].sum() / large_total
    # -----
    large_admissions = hospital_df[hospital_df.Category == "LARGE"].Yearly.sum()
    large_to_com = large_admissions * large_community_proportion
    large_to_stach = large_admissions * large_stach_proportion
    large_to_unc = large_to_stach * parameters["nonunc_to_unc"]
    large_to_large = large_to_stach * ((1 - parameters["nonunc_to_unc"]) * parameters["large_to_large"])
    large_to_small = large_to_stach * ((1 - parameters["nonunc_to_unc"]) * parameters["large_to_small"])
    large_to_ltach = large_admissions * large_lt_proportion
    large_to_nh = large_admissions * large_nh_proportion
    large_to_death = large_admissions * large_dead_proportion
    # SMALL ------------------------------------------------------------------------------------------------------------
    small_community_proportion = small_breakdown.loc[community_columns, "Total"].sum() / small_total
    small_stach_proportion = small_breakdown.loc[stach_columns, "Total"].sum() / small_total
    small_lt_proportion = small_breakdown.loc[lt_columns, "Total"].sum() / small_total
    small_nh_proportion = small_breakdown.loc[nh_columns, "Total"].sum() / small_total
    small_dead_proportion = small_breakdown.loc[death_columns, "Total"].sum() / small_total
    # -----
    small_admissions = hospital_df[hospital_df.Category == "SMALL"].Yearly.sum()
    small_to_com = small_admissions * small_community_proportion
    small_to_stach = small_admissions * small_stach_proportion
    small_to_unc = small_to_stach * parameters["nonunc_to_unc"]
    small_to_large = small_to_stach * ((1 - parameters["nonunc_to_unc"]) * parameters["small_to_large"])
    small_to_small = small_to_stach * ((1 - parameters["nonunc_to_unc"]) * parameters["small_to_small"])
    small_to_ltach = small_admissions * small_lt_proportion
    small_to_nh = small_admissions * small_nh_proportion
    small_to_death = small_admissions * small_dead_proportion
    # LTACH ----------------------------------------------------------------------------------------------------------------
    # Assumption: LTACH admissions = SUM(STACH TO LTACH discharges)
    stach_admissions = hospital_df.Yearly.sum()
    lt_admissions = unc_to_ltach + small_to_ltach + large_to_ltach
    lt_to_stach = lt_admissions * parameters["lt_to_stach"]
    lt_to_unc = lt_to_stach * (unc_to_ltach / lt_admissions)
    lt_to_large = lt_to_stach * (large_to_ltach / lt_admissions)
    lt_to_small = lt_to_stach * (small_to_ltach / lt_admissions)
    lt_to_ltach = 0
    lt_to_nh = lt_admissions * parameters["lt_to_nh"]
    lt_to_death = lt_admissions * parameters["lt_death"]
    lt_to_com = lt_admissions - lt_to_stach - lt_to_nh - lt_to_death
    # NH -------------------------------------------------------------------------------------------------------------------
    # Assumption: Equal admissions and discharges for NHs
    nh_admissions = parameters["community_to_nh"] + unc_to_nh + large_to_nh + small_to_nh + lt_to_nh
    nh_to_com = nh_admissions * ((1 - parameters["nh_death"]) * parameters["nh_to_com"])
    nh_to_death = nh_admissions * parameters["nh_death"]
    nh_to_stach = nh_admissions - nh_to_com - nh_to_death
    nh_to_unc = nh_to_stach * (unc_admissions / stach_admissions)
    nh_to_large = nh_to_stach * (large_admissions / stach_admissions)
    nh_to_small = nh_to_stach * (small_admissions / stach_admissions)
    nh_to_ltach = 0  # NH to LT is not allowed: Team Assumption
    nh_to_nh = 0  # NH to NH is not allowed: Team Assumption
    # COMMUNITY ------------------------------------------------------------------------------------------------------------
    com_to_com = 0
    com_to_unc = unc_admissions - unc_to_unc - large_to_unc - small_to_unc - lt_to_unc - nh_to_unc
    com_to_large = large_admissions - unc_to_large - large_to_large - small_to_large - lt_to_large - nh_to_large
    com_to_small = small_admissions - unc_to_small - large_to_small - small_to_small - lt_to_small - nh_to_small
    com_to_ltach = 0
    com_to_nh = parameters["community_to_nh"]
    com_to_death = 0
    # Make 6x6 -------------------------------------------------------------------------------------------------------------
    r1 = [com_to_com, com_to_unc, com_to_large, com_to_small, com_to_ltach, com_to_nh, com_to_death]
    r2 = [unc_to_com, unc_to_unc, unc_to_large, unc_to_small, unc_to_ltach, unc_to_nh, unc_to_death]
    r3 = [large_to_com, large_to_unc, large_to_large, large_to_small, large_to_ltach, large_to_nh, large_to_death]
    r4 = [small_to_com, small_to_unc, small_to_large, small_to_small, small_to_ltach, small_to_nh, small_to_death]
    r5 = [lt_to_com, lt_to_unc, lt_to_large, lt_to_small, lt_to_ltach, lt_to_nh, lt_to_death]
    r6 = [nh_to_com, nh_to_unc, nh_to_large, nh_to_small, nh_to_ltach, nh_to_nh, nh_to_death]
    sbs_columns = ["COMMUNITY", "UNC", "LARGE", "SMALL", "LT", "NH", "Death"]
    sbs = pd.DataFrame([r1, r2, r3, r4, r5, r6], columns=sbs_columns)
    sbs.index = sbs_columns[:-1]
    print("Six-by-six:")
    print(sbs)
    # TESTS: The 6-by-6 must be aligned. Equal numbers in and out for all 6 locations
    assert np.isclose(
        sbs[["COMMUNITY", "Death"]].sum().sum(), sbs.loc["COMMUNITY"].sum(), rtol=0.01
    ), "6-by-6 is not aligned for community."
    assert np.isclose(sbs[["UNC"]].sum().sum(), sbs.loc["UNC"].sum(), rtol=0.01), "6-by-6 is not aligned for UNC"
    assert np.isclose(sbs[["LARGE"]].sum().sum(), sbs.loc["LARGE"].sum(), rtol=0.01), "6-by-6 is not aligned for LARGE"
    assert np.isclose(sbs[["SMALL"]].sum().sum(), sbs.loc["SMALL"].sum(), rtol=0.01), "6-by-6 is not aligned for SMALL"
    assert np.isclose(sbs[["LT"]].sum().sum(), sbs.loc["LT"].sum(), rtol=0.01), "6-by-6 is not aligned for LTACH"
    assert np.isclose(sbs[["NH"]].sum().sum(), sbs.loc["NH"].sum(), rtol=0.01), "6-by-6 is not aligned for NHs"

    # ------------------------------------------------------------------------------------------------------------------
    # File #1: Community -----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Account for Additional Data:
    # We are not using readmission at this time: 9.4% of people discharged to community are readmitted w/in 30 days
    unc_discharges["From Comm - Adjusted"] = (
        unc_discharges["Total"] / unc_discharges["Total"].sum() * sbs.loc["COMMUNITY"]["UNC"]
    )  # To account for readmission use: - sbs.loc["COMMUNITY"]["UNC"] - sbs.loc["UNC"]["COMMUNITY"] * parameters["readmission"] * (350 / 365)
    large_discharges["From Comm - Adjusted"] = (
        large_discharges["Total"] / large_discharges["Total"].sum() * sbs.loc["COMMUNITY"]["LARGE"]
    )
    small_discharges["From Comm - Adjusted"] = (
        small_discharges["Total"] / small_discharges["Total"].sum() * sbs.loc["COMMUNITY"]["SMALL"]
    )

    # Community
    community = pd.DataFrame(
        [[county, age_group] for county in counties for age_group in age_groups], columns=["County_Code", "Age_Group"]
    ).set_index(["County_Code", "Age_Group"])
    community["Population"] = syn_pop_counts
    community["COMMUNITY"] = 0
    community["to_unc"] = distribute_discharges(unc_discharges, unc_L50, unc_50_64, unc_G65)
    community["to_large"] = distribute_discharges(large_discharges, large_L50, large_50_64, large_G65)
    community["to_small"] = distribute_discharges(small_discharges, small_L50, small_50_64, small_G65)
    community["UNC"] = community["to_unc"] / community["Population"] / 365
    community["LARGE"] = community["to_large"] / community["Population"] / 365
    community["SMALL"] = community["to_small"] / community["Population"] / 365
    community["LT"] = 0
    g65 = community[community.index.get_level_values(1) == 2].Population.sum()
    community["NH"] = [0, 0, parameters["community_to_nh"] / g65 / 365] * 100
    community["Probability"] = community[sbs_columns[:-1]].sum(axis=1)
    community = community[["Probability"] + sbs_columns[:-1]]

    # ----------------------------------------------------------------------------------------------------------------------
    # File #2:  Location Transitions ---------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    all_lists = []
    # UNC
    unc_pop = pop[pop["UNC_Discharges"] > 0].Population.sum()
    unc_and_large_pop = pop[(pop["UNC_Discharges"] > 0) & (pop["LARGE_Discharges"] > 0)].Population.sum()
    unc_and_small_pop = pop[(pop["UNC_Discharges"] > 0) & (pop["SMALL_Discharges"] > 0)].Population.sum()
    p_unc = 1
    p_large = unc_and_large_pop / unc_pop
    p_small = unc_and_small_pop / unc_pop
    for hospital in [item for item in unc_breakdown.columns if item != "Total"]:
        all_lists.extend(stach_transitions(hospital, unc_breakdown, "UNC", p_unc, p_large, p_small))
    # LARGE
    large_pop = pop[pop["LARGE_Discharges"] > 0].Population.sum()
    large_and_unc_pop = pop[(pop["LARGE_Discharges"] > 0) & (pop["UNC_Discharges"] > 0)].Population.sum()
    large_and_small_pop = pop[(pop["LARGE_Discharges"] > 0) & (pop["SMALL_Discharges"] > 0)].Population.sum()
    p_unc = large_and_unc_pop / large_pop
    p_large = 1
    p_small = large_and_small_pop / large_pop
    for hospital in [item for item in large_breakdown.columns if item != "Total"]:
        all_lists.extend(stach_transitions(hospital, large_breakdown, "LARGE", p_unc, p_large, p_small))
    # SMALL
    small_pop = pop[pop["SMALL_Discharges"] > 0].Population.sum()
    small_and_unc_pop = pop[(pop["SMALL_Discharges"] > 0) & (pop["UNC_Discharges"] > 0)].Population.sum()
    small_and_large_pop = pop[(pop["SMALL_Discharges"] > 0) & (pop["LARGE_Discharges"] > 0)].Population.sum()
    p_unc = small_and_unc_pop / small_pop
    p_large = small_and_large_pop / small_pop
    p_small = 1
    for hospital in [item for item in small_breakdown.columns if item != "Total"]:
        all_lists.extend(stach_transitions(hospital, small_breakdown, "SMALL", p_unc, p_large, p_small))
    # NH
    nh_movement = nh_to_com + nh_to_unc + nh_to_large + nh_to_small
    nh_to_com_proportion = nh_to_com / nh_movement
    nh_to_unc_proportion = nh_to_unc / nh_movement
    nh_to_large_proportion = nh_to_large / nh_movement
    nh_to_small_proportion = nh_to_small / nh_movement
    nh_to_stach_proportion = nh_to_unc_proportion + nh_to_large_proportion + nh_to_small_proportion
    for county_code in counties:
        for age_group in age_groups:
            if age_group < 2:
                row = [county_code, age_group, "NH", 1, 0, 0, 0, 0, 0]
            else:
                if county_code in unc_counties:
                    temp_unc = nh_to_unc_proportion * total_population / unc_population
                else:
                    temp_unc = 0
                if county_code in large_counties:
                    temp_large = nh_to_large_proportion * total_population / large_population
                else:
                    temp_large = 0
                stach_remaining = nh_to_stach_proportion - temp_unc - temp_large
                temp_small = 0
                if county_code in small_counties:
                    temp_small = stach_remaining
                else:
                    temp_large = nh_to_stach_proportion - temp_unc
                row = [county_code, age_group, "NH", nh_to_com_proportion, temp_unc, temp_large, temp_small, 0, 0]
            all_lists.append(row)
    # LT
    lt_movement = lt_to_com + lt_to_unc + lt_to_large + lt_to_small + lt_to_nh
    lt_to_com_p = lt_to_com / lt_movement
    lt_to_unc_proportion = lt_to_unc / lt_movement
    lt_to_large_proportion = lt_to_large / lt_movement
    lt_to_small_proportion = lt_to_small / lt_movement
    lt_to_lt_proportion = 0
    lt_to_nh_p = lt_to_nh / lt_movement
    for county in counties:
        for age_group in age_groups:
            # We don't allow <50 to go to LTACH
            if age_group == 0:
                row = [county, age_group, "LT", 1, 0, 0, 0, 0, 0]
            else:
                # All lt_to_nh_p must be contained by 65+, <65 gets a 0.
                if age_group == 2:
                    temp_nh = (1 / parameters["lt_65p"]) * lt_to_nh_p
                else:
                    temp_nh = 0
                # Increase chance of going to UNC for people from UNC catchment area
                if county in unc_counties:
                    temp_unc = lt_to_unc_proportion * total_population / unc_population
                else:
                    temp_unc = 0
                if county in large_counties:
                    temp_large = lt_to_large_proportion * total_population / large_population
                else:
                    temp_large = 0
                stach_remaining = parameters["lt_to_stach"] - temp_unc - temp_large
                temp_small = stach_remaining
                # All counties are "small" counties except for 1, so send all of them to large
                if county not in small_counties:
                    temp_small = 0
                    temp_large = parameters["lt_to_stach"] - temp_unc
                remaining_p = 1 - temp_unc - temp_large - temp_small - lt_to_lt_proportion - temp_nh
                row = [
                    county,
                    age_group,
                    "LT",
                    remaining_p,
                    temp_unc,
                    temp_large,
                    temp_small,
                    lt_to_lt_proportion,
                    temp_nh,
                ]
            all_lists.append(row)

    # Tests: All rows must add to 1.
    for row in all_lists:
        assert np.isclose(
            sum(row[3:]), 1, rtol=0.001
        ), f"Error with row {row}!. It sums to {round(sum(row[3:]), 5)}, not 1"

    temp_df = pd.DataFrame(
        all_lists, columns=["County_Code", "Age_Group", "Facility", "COMMUNITY", "UNC", "LARGE", "SMALL", "LT", "NH"]
    )

    # TO UNC Testing ---------------------------------------------------------------------------------------------------
    # No one should be able to go to UNC from a nonUNC catchment county
    assert temp_df[~temp_df.County_Code.isin(unc_counties)].UNC.sum() == 0
    # No one should be able to go to a LARGE from a nonLARGE catchment county
    assert temp_df[~temp_df.County_Code.isin(large_counties)].LARGE.sum() == 0
    # LTACH TESTING ----------------------------------------------------------------------------------------------------
    lt_df = temp_df[(temp_df.Facility == "LT") & (temp_df.Age_Group != 0)]
    gb = lt_df.groupby("Age_Group")["COMMUNITY"].mean()
    p65 = parameters["lt_65p"]
    # The LTACH to Community Probability must average X%
    assert np.isclose(gb.loc[1] * (1 - p65) + gb.loc[2] * p65, lt_to_com_p, rtol=0.01)
    # The LTACH to TACH probabilities must sum to X%
    assert np.isclose(lt_df[["UNC", "LARGE", "SMALL"]].sum(axis=1).mean(), parameters["lt_to_stach"], rtol=0.01)
    # TO NH is not allowed for Age != 2
    assert np.isclose(temp_df[temp_df.Facility == "LT"].groupby("Age_Group").sum().loc[[0, 1]].NH.sum(), 0, rtol=0)
    # NH TESTING -------------------------------------------------------------------------------------------------------
    nh_df = temp_df[(temp_df.Facility == "NH") & (temp_df.Age_Group == 2)]
    assert nh_df.NH.sum() == 0  # NH to NH is impossible
    assert nh_df.LT.sum() == 0  # NH to LT is impossible
    assert nh_df.COMMUNITY.value_counts().shape[0] == 1  # All community values are equal

    # Save Files
    sbs.to_csv(filepaths["six_by_six"]["path"])
    community.to_csv(community_transitions_path)
    temp_df.to_csv(location_transitions_path, index=False)

    # Update hospital file
    hospitals = hospital_beds[["Category", "County", "Total Beds"]].copy()
    hospitals[["Acute Beds", "ICU Beds"]] = hospital_df[["Acute Beds", "ICU Beds"]]
    hospitals["Total Beds"] = hospitals["Acute Beds"] + hospitals["ICU Beds"]
    hospitals["Acute_Covid_Agents"] = hospital_df["All COVID-19 Non-ICU Patients in Hospital"]
    hospitals["Acute_NonCovid_Agents"] = hospital_df["Acute Agents"]
    hospitals["ICU_Covid_Agents"] = hospital_df["Adult ICU Covid-19 Positive Patients"]
    hospitals["ICU_NonCovid_Agents"] = hospital_df["ICU Agents"]
    hospitals["NonCovid_Agents"] = hospital_df["Acute Agents"] + hospital_df["ICU Agents"]
    hospitals.to_csv(filepaths["hospitals"]["path"])

    # Steady States
    items = []
    to_loc = ["STEADY", "STEADY", "STEADY", "STEADY", "STEADY"]
    from_loc = ["UNC", "LARGE", "SMALL", "LT", "NH"]
    for item in ["UNC", "LARGE", "SMALL"]:
        temp = hospitals[hospitals.Category == item]
        items.append(temp["NonCovid_Agents"].sum())
    # Targets file
    items.append(parameters["n_beds_ltachs"] * parameters["ltach_fill_p"])
    items.append(parameters["n_beds_nhs"] * parameters["nh_fill_p"])
    for index in sbs.index:
        row = sbs.loc[index]
        for item in ["COMMUNITY", "UNC", "LARGE", "SMALL", "LT", "NH"]:
            from_loc.append(index)
            to_loc.append(item)
            items.append(row[item])
    targets = pd.DataFrame([from_loc, to_loc, items]).T
    targets.columns = ["From", "To", "Target"]
    for item in ["UNC", "LARGE", "SMALL", "LT", "NH"]:
        targets.loc[targets.shape[0]] = [item, "Death", sbs.loc[item, "Death"]]
    targets.to_csv(filepaths["location_targets"]["path"], index=False)
