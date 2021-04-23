from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FILEPATH_YAML = "model_runs/config/filepaths.yaml"
filepaths = yaml.load(Path(FILEPATH_YAML).read_text(), Loader=yaml.loader.SafeLoader)

# Read data
if __name__ == "__main__":
    """
    Update available bed counts with new bed counts, if available
    Otherwise, use default values
    """

    beds = pd.read_csv("./data/hospital_beds/hospitals_default.csv").set_index("Name")
    beds["Total Beds"] = beds["Beds"]

    if Path("./data/hospital_beds/sensitive/operational_beds.csv").exists():
        new_acute = pd.read_csv("./data/hospital_beds/sensitive/operational_beds.csv").drop("County", axis=1)
        new_icu = pd.read_csv("./data/hospital_beds/sensitive/ds001003-hospitalbeds-date-lookup.csv").drop(
            "County", axis=1
        )

        crosswalk = pd.read_csv("./data/hospital_beds/sensitive/hospital_crosswalk.csv")

        # drop extraneous rows from additional data
        new_icu = new_icu[new_icu.Hospital.notna()][["Hospital", "ICU Licensed Beds"]]
        new_acute = new_acute[new_acute["Primary Name"].notna()]
        # Merge together
        new_data = crosswalk.merge(new_acute, how="left", left_on="Operational", right_on="Primary Name").drop(
            "Primary Name", axis=1
        )
        new_data = new_data.merge(new_icu, how="left", left_on="ds001003", right_on="Hospital").drop("Hospital", axis=1)
        new_data = new_data.set_index("Name")
        beds = beds.merge(new_data, left_index=True, right_index=True)

        beds["Total Beds"] = beds[["Operational Acute Beds", "Beds"]].apply(
            lambda x: x["Operational Acute Beds"] if not np.isnan(x["Operational Acute Beds"]) else x["Beds"], axis=1
        )

    beds = beds.fillna(0)
    beds = beds[beds.Include == 1]
    beds = beds.drop("Johnston Health Smithfield")

    # Add the Category of the hospital
    category = []
    for _, row in beds.iterrows():
        if row.UNC == 1:
            category.append("UNC")
        else:
            if row["Total Beds"] < 400:
                category.append("SMALL")
            else:
                category.append("LARGE")
    beds["Category"] = category

    # Tests: All hospitals that are included have at least 1 bed
    assert all(beds["Total Beds"] > 0), "Not all hospitals have at least 1 bed"

    if "ICU Licensed Beds" in beds.columns:
        beds["Acute Beds"] = beds["Total Beds"] - beds["ICU Licensed Beds"]
        beds["ICU Beds"] = beds["ICU Licensed Beds"]
    else:
        beds["Acute Beds"] = beds["Total Beds"] - beds["ICU Beds"]

    beds.to_csv(filepaths["hospital_beds"]["path"])
