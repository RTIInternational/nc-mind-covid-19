from pathlib import Path

import pandas as pd
import yaml

import src.data_input as di

FILEPATH_YAML = "model_runs/config/filepaths.yaml"
filepaths = yaml.load(Path(FILEPATH_YAML).read_text(), Loader=yaml.loader.SafeLoader)


if __name__ == "__main__":
    crosswalk = di.hospital_info()
    county_codes = di.county_codes()[["County", "County_Code"]]

    # ----- File #1: Discharges by Hospital/County ---------------------------------------------------------------------
    county_discharges = pd.read_csv("data/transitions/from_pdf/2018/2018master_ptorg_final.csv")
    county_discharges = county_discharges.fillna(0).rename(columns={"RESIDENCE": "County"})
    drops = county_discharges[
        county_discharges.County.isin(
            [
                "Actual",
                "Calculated",
                "Unreported",
                "TENNESSEE",
                "SOUTH CAROLINA",
                "VIRGINIA",
                "Other/Missing",
                "GEORGIA",
            ]
        )
    ]
    county_discharges = county_discharges.drop(drops.index).reset_index(drop=True)

    # Name must be linked:
    assert len([item for item in county_discharges.columns[1:] if item not in crosswalk["ptorg name"].values]) == 0

    # Convert names to standard convention
    county_discharges.columns = [county_discharges.columns[0]] + [
        crosswalk[crosswalk["ptorg name"] == i]["Name"].values[0] for i in county_discharges.columns[1:]
    ]

    # Save Cleaned File
    county_discharges["County_Code"] = county_codes.set_index("County").loc[county_discharges["County"]].values
    county_discharges = county_discharges.drop("County", axis=1)
    county_discharges.to_csv(filepaths["county_discharges"]["path"], index=False)

    # ----- File 2: Age/Disposition Breakdown --------------------------------------------------------------------------
    demo = pd.read_csv("data/transitions/from_pdf/2018/2018_subset_ptchar_for_analysis.csv").fillna(0)
    demo = demo.rename(columns={"Unnamed: 0": "Category"})

    demo = demo.drop(["Actual", "Calculated", "Difference"], axis="columns")

    order = [
        "Patient Residence State NC",
        "Patient Residence State Not NC",
        "Age Group Less than 1 Year",
        "Age Group 1 - 17 years",
        "Age Group 18 - 44 years",
        "Age Group 45 - 64 years",
        "Age Group 65 - 84 years",
        "Age Group 85 or more years",
        "Patient Disposition Home, self, or outpatient care",
        "Patient Disposition Discharged, transferred to acute facility",
        "Patient Disposition Discharged, transferred to facility that provides nursing, custodial, or supportive care",
        "Patient Disposition Discharged, transferred",
        "Patient Disposition Discharged, transferred to long term acute care facility (LTAC)",
        "Patient Disposition Discharged, transferred to psychiatric facility",
        "Patient Disposition Hospice",
        "Patient Disposition Left against medical advice",
        "Patient Disposition Court/Law Enforcement",
        "Patient Disposition Expired",
        "Patient Disposition Other/Unknown",
    ]
    demo = demo.set_index("Category").loc[order]
    demo = demo.reset_index()

    # Names must be in standard convention
    assert len([item for item in demo.columns[1:] if item not in crosswalk["Name"].values]) == 0

    demo.to_csv(filepaths["sheps_demographics"]["path"], index=False)
