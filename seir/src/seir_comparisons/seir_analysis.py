"""
# ----- Assess the SEIR model's output at varying levels of updates.
# 1: A global Re value
# 2: Introduction of county specific Re values
# 3: Introducing S and I updates based on antibody tests
# 4:
"""

from seir.src.seir_comparisons.simple_seir import prepare_data, seir
from seir.src.create_county_metrics import add_re_variables, add_metrics
import pandas as pd
import tqdm
import plotly.express as px


rE = 1.3
time_limit = 30

descriptions = {
    0: "CC: False, CM: 10, ABC: False",
    1: "CC: True, CM: 10, ABC: False",
    2: "CC: True, CM: 6, ABC: False",
    3: "CC: True, CM:6, ABC: True",
}


def run_seir(rE: float, use_correction: bool = True, case_multiplier: float = 6, prs: float = None) -> pd.DataFrame:
    """Run county specific SEIR models

    Args:
        rE (float): The statewide average r0 to use for the models
        use_correction (bool): Should we account for difference in past COVID-19 growth of counties?
        case_multiplier (float): The number of cases to multiply reported cases by
        prs (float): The percent remaining susceptible when the model starts

    Returns:
        pd.DataFrame: Dataframe with expected cases by day for each county
    """
    df_dict, pop_dict = prepare_data(case_multiplier)
    df_dict = add_re_variables(df_dict)
    correction_df = add_metrics(df_dict, pop_dict)

    # Use 4/24 as an estimate for when strict stay at home orders took effect.
    r0_estimate = df_dict["North Carolina"].loc["2020-04-24"]["Rolling_Avg_Re"]

    final_df = pd.DataFrame()
    run_range = tqdm.trange(len(df_dict.keys()), desc="---> Running SEIR Models for Each County")
    for i in run_range:
        item = list(df_dict.keys())[i]
        temp_df = df_dict[item]
        time_start = temp_df.Day.max() + 1

        temp_re = rE
        if use_correction:
            temp_re = rE * correction_df.loc[item].re_correction
        seir_df = seir(
            df=temp_df,
            time_start=time_start,
            time_limit=time_limit,
            rE=temp_re,
            r0_estimate=r0_estimate,
            item_population=pop_dict[item],
            case_multiplier=case_multiplier,
            percent_remaining_susceptible=prs,
            item_name=item,
        )

        final_df = pd.concat([final_df, seir_df]).reset_index(drop=True)

    return final_df


# --- Flu Regions ------------------------------------------------------------------------------------------------------
flu_regions = pd.read_csv("data/flu_regions.csv")
flu_regions.region = flu_regions.region.astype(str).apply(lambda x: f"Flu Region {x}")
flu_regions_dict = dict(zip(flu_regions.County, flu_regions.region))
flu_regions_dict["North Carolina"] = "North Carolina"


# --- Reality ----------------------------------------------------------------------------------------------------------
data = pd.read_csv("seir/src/seir_comparisons/cases_by_day.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.rename(columns={"New cases": "Reported Cases", "Date": "Day"})
data["Flu Regions"] = data["County"].apply(lambda x: flu_regions_dict[x])
# Get Flu Regions
reality = data.groupby(by=["Flu Regions", "Day"]).sum().reset_index()
# Get NC
nc = reality.groupby(by="Day").sum().reset_index()
nc.insert(0, "Flu Regions", "North Carolina")
reality = reality.append(nc)


# ----- Run the various runs -------------------------------------------------------------------------------------------
r1 = run_seir(rE, use_correction=False, case_multiplier=10, prs=None)
r2 = run_seir(rE, use_correction=True, case_multiplier=10, prs=None)
r3 = run_seir(rE, use_correction=True, case_multiplier=6, prs=None)
r4 = run_seir(rE, use_correction=True, case_multiplier=6, prs=0.9)


# ----- Aggregate by the flu region ------------------------------------------------------------------------------------
runs = [r1, r2, r3, r4]
agg_runs = [pd.DataFrame()] * 4

for _, run_df in enumerate(runs):
    run_df["Flu Region"] = run_df["County"].apply(lambda x: flu_regions_dict[x])
    agg_runs[_] = run_df.groupby(by=["Flu Region", "Day"]).sum().reset_index()


# ----- Create the visualizations --------------------------------------------------------------------------------------
for region in agg_runs[0]["Flu Region"].unique():
    cdf = pd.DataFrame()
    for _, x in enumerate(agg_runs):
        temp_df = x[x["Flu Region"] == region][-time_limit:][["reported_cases", "Day"]]
        temp_df["SEIR Model"] = descriptions[_]
        temp_df.columns = ["Reported Cases", "Day", "SEIR Model Description"]
        cdf = cdf.append(temp_df).reset_index(drop=True)

    # Add Reality
    temp_reality = reality[reality["Flu Regions"] == region][["Reported Cases", "Day"]]
    temp_reality["SEIR Model Description"] = "Actual Reported Cases"
    rolling = temp_reality["Reported Cases"].rolling(7).mean()
    temp_reality["Reported Cases"] = rolling
    temp_reality = temp_reality[temp_reality.Day > "11-01-2020"]

    cdf = cdf.append(temp_reality)

    fig = px.line(
        cdf,
        x="Day",
        y="Reported Cases",
        color="SEIR Model Description",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title=f"Comparing Actual Reported Cases to SEIR Estimates For: {region}",
    )
    fig.write_html(f"seir/src/seir_comparisons/output/{region}.html")

# ----- Calculate the accuracy -----------------------------------------------------------------------------------------


# ----- Graphic for Powerpoint -----------------------------------------------------------------------------------------
s10 = r1[(r1["County"] == "North Carolina") & (r1.Day < "11-16-2020")].copy()
s10["Remaining Susceptible"] = (10500000 - s10["cumulative_infections"]) / 10500000
s10["Case Multiplier"] = "10"

s6 = r3[(r3["County"] == "North Carolina") & (r3.Day < "11-16-2020")].copy()
s6["Remaining Susceptible"] = (10500000 - s6["cumulative_infections"]) / 10500000
s6["Case Multiplier"] = "6"

ex_df = pd.DataFrame()
ex_df = ex_df.append(s10[["Day", "Remaining Susceptible", "Case Multiplier"]])
ex_df = ex_df.append(s6[["Day", "Remaining Susceptible", "Case Multiplier"]])

fig = px.line(
    ex_df,
    x="Day",
    y="Remaining Susceptible",
    color="Case Multiplier",
    color_discrete_sequence=px.colors.qualitative.Safe,
    title="Comparing Remaining Susceptible Population for Various Case Multipliers",
)
fig.write_html("test.html")


# ----- County Re values for each region -------------------------------------------------------------------------------
df_dict, pop_dict = prepare_data(1)
df_dict = add_re_variables(df_dict)
correction_df = add_metrics(df_dict, pop_dict)
correction_df["Flu Region"] = [flu_regions_dict[x] for x in correction_df.index]
correction_df["Population"] = [pop_dict[x] for x in correction_df.index]
correction_df = correction_df.sort_values(by="Re", ascending=False)

correction_df = correction_df[["Population", "Re", "re_correction", "Flu Region"]]
correction_df.columns = ["Population", "Previous Re", "Re Correction", "Flu Region"]

for region in correction_df["Flu Region"].unique():
    temp_df = correction_df[correction_df["Flu Region"] == region].round(3)
    temp_df.to_csv(f"seir/src/seir_comparisons/csvs/{region}.csv")

    # --- Find the weighted Re Correction
    temp_df = temp_df[temp_df["Re Correction"] != 1]
    rem = (temp_df["Re Correction"] * (temp_df["Population"] / temp_df["Population"].sum())).sum()
    print(f"{region}: {rem}")


# ---- Region Re Estimates:
df_dict, pop_dict = prepare_data(1, include_flu_regions=True)
df_dict = add_re_variables(df_dict)
correction_df = add_metrics(df_dict, pop_dict)

correction_df.loc["Flu Region 4"]
correction_df.loc["Flu Region 6"]
