import pandas as pd

year = "2018"

import_file = year + "_master_ptchar_final.csv"
final = pd.read_csv(import_file)
final = final.set_index(["variable"], drop=True)

subset_rows = [
    "Patient Residence State NC",
    "Patient Residence State SC",
    "Patient Residence State VA",
    "Patient Residence State GA",
    "Patient Residence State TN",
    "Patient Residence State Other",
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

subset_df = final[final.index.isin(subset_rows)]
subset_df = subset_df.replace(".", "")
subset_df = subset_df.replace({",": ""}, regex=True).apply(pd.to_numeric, 1)
subset_df = subset_df.apply(pd.to_numeric)

# Sum other states, note that not every option occurs each time
other_states = [
    "Patient Residence State SC",
    "Patient Residence State VA",
    "Patient Residence State GA",
    "Patient Residence State TN",
    "Patient Residence State Other",
]
other_states_sum = subset_df[subset_df.index.isin(other_states)]
other_states_sum = other_states_sum.sum(axis=0)
other_states_sum = pd.DataFrame(other_states_sum).transpose()
other_states_sum.index = ["Patient Residence State Not NC"]

subset_df = subset_df.append(other_states_sum, ignore_index=False, sort=True)
subset_df = subset_df.drop(other_states)

subset_df.loc[:, "Actual"] = subset_df["Summary Data for All Hospitals"]
subset_df = subset_df.drop(columns=["Summary Data for All Hospitals"])
subset_df.loc[:, "Calculated"] = subset_df.sum(axis=1)
subset_df.loc[:, "Difference"] = subset_df["Actual"] - (subset_df["Calculated"] - subset_df["Actual"])

if abs(subset_df["Difference"]).sum() != 0:
    print("Unexpected difference between calculated in actual in one or more categories")

export_subset = year + "_subset_ptchar_for_analysis.csv"
subset_df.to_csv(export_subset)
