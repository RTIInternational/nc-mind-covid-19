"""This file is for functions that process raw data
read in by functions in data_input.py"""

from functools import lru_cache

import src.data_input as di


@lru_cache()
def sample_population(limit: int, seed: int):
    population_data = di.read_population()
    if limit < len(population_data):
        return population_data.sample(limit, random_state=seed).reset_index(drop=True)
    return population_data


def lhd_region_map():
    df = di.county_codes()
    unique_regions = df["lhd_regions"].unique()
    region_map = {region: i for i, region in enumerate(unique_regions)}
    return region_map


def flu_region_map():
    df = di.county_codes()
    unique_regions = df["flu_regions"].unique()
    region_map = {region: i for i, region in enumerate(unique_regions)}
    return region_map


def county_to_lhd_region_map():
    df = di.county_codes()
    return {row.County_Code: row.lhd_regions for row in df.itertuples()}


def county_to_flu_region_map():
    df = di.county_codes()
    return {row.County_Code: row.flu_regions for row in df.itertuples()}


def county_to_coalition_map():
    df = di.county_codes()
    return {row.County_Code: row.prep_coalitions for row in df.itertuples()}


def county_name_to_lhd_region_map():
    df = di.county_codes()
    return {row.County: row.lhd_regions for row in df.itertuples()}


def county_name_to_flu_region_map():
    df = di.county_codes()
    return {row.County: row.flu_regions for row in df.itertuples()}
