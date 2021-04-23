from typing import Tuple

import pandas as pd

from seir.src.simple_seir import prepare_data, seir_params


def add_re_variables(df_dict: dict, shift: int = 14):
    """Add the estimated rolling r effective for a specific county or the entire state

    Args:
        df_dict (dict): dictionary of dataframes by region/county/state
        shift (int): Rolling average window

    Returns:
        [type]: [description]
    """
    for key in df_dict.keys():
        cases_df = df_dict[key]
        # Average new cases over previous 7 days:
        cases_df["Avg New Cases"] = cases_df["Cases"].rolling(shift).mean().fillna(0)

        # If missing data for 30 of the last 60 days, we don't use this
        if sum(cases_df.Cases[-60:] == 0) > 30:
            for item in ["Rolling_Avg_Gr", "Rolling_Avg_Re"]:
                cases_df[item] = None
        else:
            cases_df["Rolling_Avg_Gr"] = (cases_df["Avg New Cases"] / cases_df["Avg New Cases"].shift(shift)) ** (
                1 / shift
            ) - 1
            le = 1 / seir_params["length_exposure"]
            li = 1 / seir_params["length_infection"]
            cases_df["Rolling_Avg_Re"] = cases_df["Rolling_Avg_Gr"].apply(lambda x: (1 + x / le) * (1 + x / li))
        df_dict[key] = cases_df
    return df_dict


def add_metrics(df_dict: dict, pop_dict: dict) -> Tuple[pd.DataFrame, float]:
    """ Add a bunch of different metrics

    Args:
        df_dict: dict
        pop_dict: dict

    Returns:
        Tuple[pd.DataFrame, float]: A dataframe with re correction values and the overall state re value
    """

    # Estimate the rE values
    numbers_dict = {
        "Last Reported Cases": {},
        "Cumulative Cases": {},
        "Last Estimated Infections": {},
        "Cumulative Infections": {},
        "Re": {},
        "GR": {},
        "Avg New Cases": {},
        "1 Week Ago Re": {},
        "1 Week Ago GR": {},
        "1 Week Ago Avg New Cases": {},
        "2 Weeks Ago Re": {},
        "2 Weeks Ago GR": {},
        "2 Weeks Ago Avg New Cases": {},
        "Population": {},
        "Reason Not Included": {},
    }

    for item in df_dict.keys():
        temp_df = df_dict[item]
        numbers_dict["Last Reported Cases"][item] = temp_df.Cases.values[-1]
        numbers_dict["Last Estimated Infections"][item] = temp_df["Estimated Infections"].values[-1]
        numbers_dict["Cumulative Cases"][item] = temp_df.CumTot.values[-1]
        numbers_dict["Cumulative Infections"][item] = temp_df["Cumulative Infections"].values[-1]
        numbers_dict["Population"][item] = pop_dict[item]

        calculate = True
        if temp_df["Rolling_Avg_Re"].values[0] is None:
            calculate = False
            numbers_dict["Reason Not Included"][item] = "Did not consistently report data."
        elif any(temp_df["Avg New Cases"].values[-7:] < 5):
            calculate = False
            numbers_dict["Reason Not Included"][item] = "Averaged less than 5 cases."

        if calculate:
            numbers_dict["Re"][item] = temp_df["Rolling_Avg_Re"].values[-1]
            numbers_dict["1 Week Ago Re"][item] = temp_df["Rolling_Avg_Re"].values[-8]
            numbers_dict["2 Weeks Ago Re"][item] = temp_df["Rolling_Avg_Re"].values[-15]
            numbers_dict["GR"][item] = temp_df["Rolling_Avg_Gr"].values[-1]
            numbers_dict["1 Week Ago GR"][item] = temp_df["Rolling_Avg_Gr"].values[-8]
            numbers_dict["2 Weeks Ago GR"][item] = temp_df["Rolling_Avg_Gr"].values[-15]
            numbers_dict["Avg New Cases"][item] = temp_df["Avg New Cases"][-1]
            numbers_dict["1 Week Ago Avg New Cases"][item] = temp_df["Avg New Cases"][-8]
            numbers_dict["2 Weeks Ago Avg New Cases"][item] = temp_df["Avg New Cases"][-15]
            numbers_dict["Reason Not Included"][item] = ""
        else:
            for key in [
                "Re",
                "1 Week Ago Re",
                "2 Weeks Ago Re",
                "GR",
                "1 Week Ago GR",
                "2 Weeks Ago GR",
                "Avg New Cases",
                "1 Week Ago Avg New Cases",
                "2 Weeks Ago Avg New Cases",
            ]:
                numbers_dict[key][item] = None

    df = pd.DataFrame.from_dict(numbers_dict)
    state_re = df.loc["North Carolina"].Re
    df["re_correction"] = [item ** (1 / 3) / state_re ** (1 / 3) if item > 0 else 1 for item in df.Re]
    return df


if __name__ == "__main__":
    """ Calculate the county specific re correction values
    """
    df_dict, pop_dict = prepare_data(include_lhd_regions=True, include_flu_regions=True)
    df_dict = add_re_variables(df_dict)
    df = add_metrics(df_dict, pop_dict)
    df = df.reset_index()
    df = df.rename(columns={"index": "Region"})
    df = df[(df.Region.str.contains("Flu")) | (df.Region == "North Carolina")]
    df = df.loc[df.Region.sort_values().index]

    # Clean-up
    for item in [
        "Re",
        "1 Week Ago Re",
        "2 Weeks Ago Re",
        "Avg New Cases",
        "1 Week Ago Avg New Cases",
        "2 Weeks Ago Avg New Cases",
        "re_correction",
    ]:
        df[item] = df[item].round(2)
    for item in ["Last Reported Cases", "Last Estimated Infections", "Cumulative Cases", "Cumulative Infections"]:
        df[item] = df[item].astype(int)
    for item in ["GR", "1 Week Ago GR"]:
        df[item] = df[item].astype(float).map("{:.2%}".format)

    # Save file for state excel
    (
        df.drop(
            [
                "re_correction",
                "2 Weeks Ago Re",
                "2 Weeks Ago GR",
                "Population",
                "Reason Not Included",
                "Last Estimated Infections",
                "Cumulative Infections",
            ],
            axis=1,
        ).to_excel("seir/region_level_values.xlsx", index=False)
    )

    # Save file for exploring
    tsdf = pd.concat(df_dict, axis=0, sort=False).reset_index().rename(columns={"level_0": "County"})
    tsdf.to_csv("seir/visual/time_series_data.csv")
