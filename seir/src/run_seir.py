import pandas as pd
import numpy as np
import datetime as dt
import os

import tqdm
from seir.src.create_county_metrics import add_re_variables, add_metrics
from seir.src.simple_seir import prepare_data, seir


def run_seir(
    rE: float,
    use_correction: bool = True,
    use_county_re: bool = False,
    case_multiplier: float = 6,
    percent_remaining_susceptible: float = 0.90,
    time_limit: int = 30,
    print_status: bool = True,
    just_regions: bool = False,
) -> pd.DataFrame:
    """Run county specific SEIR models

    Args:
        rE (float): The statewide average r0 to use for the models
        use_correction (bool): Should we account for difference in past COVID-19 growth of counties?
        use_county_re (bool): Should we use the most recent rE calculated for a county?
        case_multiplier (float): The number of cases to multiply reported cases by
        time_limit (int): Number of days to run the SEIR model

    Returns:
        pd.DataFrame: Dataframe with expected cases by day for each county
    """
    # See if rE value has been ran for the current day:
    rE = round(rE, 3)
    file_name = "re_{}_cm_{}_{}".format(rE, case_multiplier, str(dt.datetime.today().date()) + ".csv")
    if file_name in os.listdir("seir/"):
        df = pd.read_csv("seir/" + file_name)
        df.Day = pd.to_datetime(df.Day)
        return df

    df_dict, pop_dict = prepare_data(case_multiplier)
    df_dict = add_re_variables(df_dict)
    correction_df = add_metrics(df_dict, pop_dict)

    # Estimate r0 for the state: If no data, use 1.25 as an estimate (this has very little impact on SEIR results)
    try:
        # Use 4/24 as an estimate for when strict stay at home orders took effect.
        r0_estimate = df_dict["North Carolina"].loc["2020-04-24"]["Rolling_Avg_Re"]
    except Exception as E:
        print(E)
        r0_estimate = 1.25

    final_df = pd.DataFrame()
    run_range = range(len(df_dict.keys()))
    if print_status:
        run_range = tqdm.trange(len(df_dict.keys()), desc="---> Running SEIR Models for Each County")
    for i in run_range:
        item = list(df_dict.keys())[i]
        # Either run counties or regions, but not both
        if just_regions:
            if not isinstance(item, (int, np.int64)):
                continue
        else:
            if isinstance(item, (int, np.int64)):
                continue
        temp_df = df_dict[item]
        time_start = temp_df.Day.max() + 1

        temp_re = rE
        if use_correction:
            temp_re = rE * correction_df.loc[item].re_correction
        if use_county_re:
            temp_re = correction_df.loc[item].Re
        seir_df = seir(
            df=temp_df,
            time_start=time_start,
            time_limit=time_limit + 5,
            rE=temp_re,
            r0_estimate=r0_estimate,
            item_population=pop_dict[item],
            case_multiplier=case_multiplier,
            percent_remaining_susceptible=percent_remaining_susceptible,
            item_name=item,
        )

        final_df = pd.concat([final_df, seir_df]).reset_index(drop=True)

    # Build a report:
    county_pop = pd.read_csv("seir/county_values.csv")
    not_enough_data = (correction_df.loc[county_pop.County.values].re_correction == 1).sum()
    print("---> {} counties did not have enough data to calculate an Re correction.".format(not_enough_data))
    final_df.to_csv("seir/" + file_name, index=False)
    return final_df
