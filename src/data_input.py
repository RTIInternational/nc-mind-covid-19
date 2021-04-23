import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.misc_functions import generate_distance_probability_distribution, get_inverted_distance_probabilities

FILEPATH_YAML = "model_runs/config/filepaths.yaml"
filepaths = yaml.load(Path(FILEPATH_YAML).read_text(), Loader=yaml.loader.SafeLoader)


def clean_discharges(df):
    hospitals = [item for item in df.columns if item not in ["County_Code", "Total"]]
    df[hospitals] = df[hospitals].div(df["Total"], axis=0).fillna(0)
    df.drop("Total", axis=1, inplace=True)
    return df


def read_population():
    pop = pd.read_parquet(filepaths["synthetic_population_file_parquet"]["path"])
    return pop


def facility_transitions() -> pd.DataFrame:
    df = pd.read_csv(filepaths["location_transitions"]["path"])
    return df


def community_transitions() -> pd.DataFrame:
    df = pd.read_csv(filepaths["community_transitions"]["path"])
    return df


def locations_by_distance() -> pd.DataFrame:
    df = pd.read_csv(filepaths["locations_by_distance"]["path"])
    return df


def county_discharges() -> pd.DataFrame:
    df = pd.read_csv(filepaths["county_discharges"]["path"]).set_index("County_Code")
    df["Johnston Health Clayton"] += df["Johnston Health Smithfield"]
    return df.drop(["Johnston Health Smithfield"], axis=1)


def demographics() -> pd.DataFrame:
    df = pd.read_csv(filepaths["sheps_demographics"]["path"])
    df["Johnston Health Clayton"] += df["Johnston Health Smithfield"]
    return df.drop(["Johnston Health Smithfield"], axis=1)


def county_hospital_distances() -> Dict[int, Any]:
    fp = Path(filepaths["county_hospital_distances_sorted"]["path"])
    county_to_hospital_distances = json.loads((fp).read_text())
    county_to_hospital_distances = {int(k): v for k, v in county_to_hospital_distances.items()}
    return county_to_hospital_distances


def nh_los() -> pd.DataFrame:
    df = pd.read_csv(filepaths["nh_los"]["path"])
    return df


def nh_los2() -> pd.DataFrame:
    df = pd.read_csv(filepaths["nh_los2"]["path"])
    return df


def hospital_beds() -> pd.DataFrame:
    df = pd.read_csv(filepaths["hospital_beds"]["path"])
    return df


def hospital_crosswalk() -> pd.DataFrame:
    df = pd.read_csv(filepaths["hospital_crosswalk"]["path"])
    return df


def hospital_info() -> pd.DataFrame:
    df = pd.read_csv(filepaths["hospital_info"]["path"])
    return df


def hospitals() -> pd.DataFrame:
    df = pd.read_csv(filepaths["hospitals"]["path"])
    return df


def nursing_homes() -> pd.DataFrame:
    df = pd.read_csv(filepaths["nursing_homes"]["path"])
    return df


def ltachs() -> pd.DataFrame:
    df = pd.read_csv(filepaths["ltachs"]["path"])
    return df


def county_facility_distribution(loc_type, distance_weight=1, bed_count_weight=1, closest_n=5) -> Dict[int, Any]:
    fp = Path(filepaths[f"county_{loc_type}_distances_sorted"]["path"])
    # read in the distances
    county_to_facility_distances = json.loads((fp).read_text())

    if loc_type == "ltach":
        bed_dict = ltachs().set_index("Name")["Beds"].to_dict()
    elif loc_type == "nh":
        bed_dict = nursing_homes().set_index("Name")["Beds"].to_dict()
    elif loc_type == "hospital":
        hos_df = hospitals()
        bed_dict = hos_df.loc[hos_df["Category"] == "LARGE"].set_index("Name")["Total Beds"].to_dict()
    else:
        print(loc_type)
        raise ValueError("ERROR: loc type invalid")

    probability_distributions = {}
    # for each county, get a dataframe of LTACH, distance, bedcount
    for county, county_list in county_to_facility_distances.items():
        county_list = [i for i in county_list if i["Name"] in bed_dict.keys()]
        county_dict = [
            {"Name": val["Name"], "beds": bed_dict[val["Name"]], "distance": val["distance_mi"]} for val in county_list
        ]
        county_df = pd.DataFrame(county_dict).set_index("Name")
        probability_distributions[int(county)] = generate_distance_probability_distribution(
            county_df, distance_weight, bed_count_weight, closest_n
        )
    return probability_distributions


def facility_to_county_probabilities(loc_type) -> Dict[int, Any]:
    # get the distance between each LTACH and each county
    fp = Path(filepaths[f"county_{loc_type}_distances_sorted"]["path"])
    county_to_facility_distances = json.loads((fp).read_text())
    # invert it to get the distances between NH and all counties
    return get_inverted_distance_probabilities(county_to_facility_distances)


@lru_cache()
def county_codes() -> pd.DataFrame:
    dtype = {
        "County": str,
        "County_Code": int,
        "county": str,
        "code": str,
        "three_regions": str,
        "lhd_regions": int,
        "prep_coalitions": str,
        "flu_regions":int,
    }
    df = pd.read_csv(filepaths["county_codes"]["path"], dtype=dtype)
    return df


def location_targets() -> pd.DataFrame:
    df = pd.read_csv(filepaths["location_targets"]["path"])
    return df
