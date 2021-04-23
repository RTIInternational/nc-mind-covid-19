import argparse as arg
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import POPULATION
from src.data_input import filepaths


def rule_of_3(df: pd.DataFrame) -> pd.DataFrame:
    """
    The rule of 3 states that if a demographic is missing from our population, adding "3" individuals of that
    demographic will have a minimal impact on our results
    """
    x = 3
    gb = df.groupby(["County_Code", "Age_Group"])
    gb = pd.DataFrame(gb.size())
    gb = gb.reset_index()

    # Go through each combination of county and age. Create a list containing the demographic
    # combination and the number of agents that are needed.
    agents_needed = []
    for county in df.County_Code.unique():
        for age_group in df.Age_Group.unique():
            temp_gb = gb[(gb.County_Code == county) & (gb.Age_Group == age_group)]
            if temp_gb.shape[0] == 0:
                agents_needed.append([county, age_group])
            elif temp_gb[0].values[0] < x:
                agents_needed.append([county, age_group, x - temp_gb[0].values[0]])

    # Add agents to the synthetic population based on which demographics were missing
    new_df = pd.DataFrame()
    print("The rule of 3 will now alter: {} demographic combinations".format(len(agents_needed)))
    for row in agents_needed:
        # row[4] contains the number of agents to append.
        for i in range(row[4]):
            # Assign a random age in years
            if row[2] == 0:
                age_years = 25
            elif row[2] == 1:
                age_years = 55
            else:
                age_years = 70
            # put the row together
            values = row[0:4] + [age_years] + row[5:]
            values = pd.DataFrame(values).T
            values.columns = df.columns
            new_df = new_df.append(values, ignore_index=True)

    return df.append(new_df, ignore_index=True)


def extract_syn_pop():
    """Extract the necessary columns from the 2017 synthetic population to use for the models"""

    # Read the 2017 Synthetic Persons and Households files

    df = pd.read_csv("data/synthetic_population/37/NC2017_Persons.csv", usecols=["hh_id", "agep", "sex", "rac1p"])

    df_household = pd.read_csv(
        "data/synthetic_population/37/NC2017_Households.csv", usecols=["hh_id", "logrecno", "county", "tract", "blkgrp"]
    )
    df = df.merge(df_household)

    df = df.rename(columns={"agep": "Age", "sex": "Sex", "rac1p": "Race", "county": "County_Code"})

    # Correct the Age
    df["Age_Years"] = df["Age"]
    df["Age_Group"] = -1
    df.loc[df["Age_Years"] < 50, "Age_Group"] = 0
    df.loc[df["Age_Years"] > 64, "Age_Group"] = 2
    df.loc[df["Age_Group"] == -1, "Age_Group"] = 1
    # Correct the Race
    df.loc[df["Race"] > 2, "Race"] = 3

    df = df[["County_Code", "Sex", "Age_Group", "Race", "Age_Years", "tract", "blkgrp", "logrecno"]]

    # Make sure every demographic has at least 3 agents.
    df = rule_of_3(df)

    # Pad the population to 10.5m people. The approximate population of NC for 2017
    number_to_add = POPULATION - df.shape[0]
    new_people = df.sample(number_to_add)
    df = df.append(new_people)
    df = df.reset_index(drop=True)
    df["Start_Location"] = 0
    df["County_Code"] = df["County_Code"].astype(int)

    # Save as parquet file
    fp = Path(filepaths["synthetic_population_file_parquet"]["path"])
    df.to_parquet(fp)


if __name__ == "__main__":
    parser = arg.ArgumentParser(description="None")

    # Set the random seed
    parser.add_argument("--seed", type=str, default=1111, help="Seed to use for the model")

    args = parser.parse_args()
    np.random.seed(args.seed)

    extract_syn_pop()
