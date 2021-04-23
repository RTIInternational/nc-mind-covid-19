import numpy as np
import pandas as pd
import datetime as dt

seir_params = {"length_infection": 6, "length_exposure": 5}


def prepare_data(case_multiplier: float = 6, include_lhd_regions: bool = False, include_flu_regions: bool = False):
    county_data = pd.read_csv("seir/src/seir_comparisons/cases_by_day.csv")
    county_data.columns = ["County", "Date", "Cases", "CumTot"]
    county_data["Date"] = pd.to_datetime(county_data["Date"])
    county_data = county_data[county_data["Date"] < "11-16-2020"]
    county_data["Estimated Infections"] = county_data["Cases"] * case_multiplier
    county_data["Cumulative Infections"] = county_data["CumTot"] * case_multiplier

    covid_start_date = county_data.Date.min().date()
    covid_end_date = county_data.Date.max().date()

    county_pop = pd.read_csv("seir/county_values.csv")

    date_list = [covid_start_date + dt.timedelta(days=x) for x in range((covid_end_date - covid_start_date).days + 1)]
    county_list = county_data.County.unique()

    df = pd.MultiIndex.from_product([date_list, [j for j in county_list]]).to_frame().reset_index(drop=True)
    df.columns = ["Date", "County"]
    df = df.set_index(["Date", "County"])

    county_data = county_data.set_index(["Date", "County"])
    county_data = df.merge(county_data, left_index=True, right_index=True, how="left")
    county_data = county_data.fillna(0)

    county_data["Day"] = (
        county_data.reset_index()["Date"].apply(lambda x: (x - pd.Timestamp(covid_start_date)).days).values
    )
    county_data = county_data.reset_index()

    # Counties
    df_dict = {}
    pop_dict = {}
    for county in county_data.County.unique():
        df_dict[county] = county_data[county_data.County == county].set_index("Date")
        df_dict[county]["CumTot"] = df_dict[county]["Cases"].cumsum()
        df_dict[county]["Cumulative Infections"] = df_dict[county]["Estimated Infections"].cumsum()
        pop_dict[county] = county_pop[county_pop["County"] == county]["Population"].values[0]

    # State
    state_df = county_data.groupby(by="Date").sum()
    state_df["CumTot"] = state_df["Cases"].cumsum()
    state_df["Day"] = [int(item / 100) for item in state_df["Day"]]
    df_dict["North Carolina"] = state_df
    pop_dict["North Carolina"] = county_pop.Population.sum()

    # Regions
    if include_lhd_regions:
        region_df = pd.read_csv("data/county_codes.csv")
        for region in region_df.lhd_regions.unique():
            temp_df = county_data[county_data.County.isin(region_df[region_df.lhd_regions == region].County)]
            temp_df = temp_df.groupby("Date").sum()
            temp_df["CumTot"] = temp_df["Cases"].cumsum()
            temp_df["Cumulative Infections"] = temp_df["Estimated Infections"].cumsum()
            temp_df["Day"] = [item for item in range(0, temp_df.shape[0])]
            region_str = "LHD Region {}".format(str(region))
            df_dict[region_str] = temp_df
            pop_dict[region_str] = county_pop[
                county_pop.County.isin(region_df[region_df.lhd_regions == region].County)
            ].Population.sum()

    # Flu Regions
    if include_flu_regions:
        flu_regions = pd.read_csv("data/flu_regions.csv")
        for region in flu_regions.region.unique():
            temp_df = county_data[county_data.County.isin(flu_regions[flu_regions.region == region].County)]
            temp_df = temp_df.groupby("Date").sum()
            temp_df["CumTot"] = temp_df["Cases"].cumsum()
            temp_df["Cumulative Infections"] = temp_df["Estimated Infections"].cumsum()
            temp_df["Day"] = [item for item in range(0, temp_df.shape[0])]
            region_str = "Flu Region {}".format(str(region))
            df_dict[region_str] = temp_df
            pop_dict[region_str] = county_pop[
                county_pop.County.isin(flu_regions[flu_regions.region == region].County)
            ].Population.sum()

    return df_dict, pop_dict


def seir(
    df,
    time_start: int,
    time_limit: int,
    r0_estimate: float,
    rE: float,
    item_population: int,
    case_multiplier: float,
    percent_remaining_susceptible: float,
    item_name: str,
) -> pd.DataFrame:
    """Run an SEIR model for COVID19. There are 3 key values:
    1. Percent remaining susceptible: On day 0, how many people are susceptible?
    2. rE: What is the level of spread?
    3. case multiplier: For each reported case, how many actual cases are there?

    Args:
        time_start (int): Days after 3/3/2020
        time_limit (int): Number of days to run the model
        r0 (float): The estimated r0
        item_population (int): Number of people who live in the county or region
        case_multiplier (float): Multiplier for underreported cases
        item_name (str): Name of the county or region

    Returns:
        [pd.DataFrame]: A data frame containing the results of the model.
    """

    # ----- Initial r0 estimate using growth rate from 3/3 to 4/24
    alpha = 1 / seir_params["length_exposure"]
    recovery_gamma = 1 / seir_params["length_infection"]
    initial_beta = r0_estimate / seir_params["length_infection"]

    # ----- SEIR vectors
    susceptible = np.full((time_limit + time_start,), 1, dtype=np.float)
    exposed = np.full((time_limit + time_start,), 0, dtype=np.float)
    infected = np.full((time_limit + time_start,), 0, dtype=np.float)
    recovered = np.full((time_limit + time_start,), 0, dtype=np.float)
    projection = np.full((time_limit + time_start,), 0, dtype=np.float)

    # ----- Incorporate Existing Data if Available
    k = int
    for k in range(1, time_start):
        moving_out_s = initial_beta * susceptible[k - 1] * infected[k - 1]
        moving_out_i = recovery_gamma * infected[k - 1]
        # If we have data, use it
        if k in df["Day"].values:
            moving_out_e = df[df["Day"] == k]["Estimated Infections"].values[0] / item_population
        else:
            moving_out_e = 0

        susceptible[k] = susceptible[k - 1] - moving_out_s
        exposed[k - 1] = moving_out_e / alpha  # How many exposed are required for day k - 1 for this level of infected?
        infected[k] = infected[k - 1] + moving_out_e - moving_out_i
        recovered[k] = recovered[k - 1] + moving_out_i

        # Number exposed on day before model begins has extreme impact on eventual cases.
        # Here we find an estimate for the number of exposed based on the last weeks worth of data
        if k == time_start - 1:
            r = df["Rolling_Avg_Gr"].values[-1]
            if not r:
                r = 1
            exposed[k] = (r + 1) * df["Avg New Cases"].values[-1] * case_multiplier / alpha / item_population

        # Make sure everything adds to 1 for k and (k - 1)
        for new_k in [k, k - 1]:
            total = susceptible[new_k] + infected[new_k] + exposed[new_k] + recovered[new_k]
            if total > 1:
                susceptible[new_k] = susceptible[new_k] - (total - 1)
            if total < 1:
                susceptible[new_k] = susceptible[new_k] + (1 - total)

    # ----- If no data for county/region, set initial cases to 1
    if df.shape[0] == 0:
        # Set initials with 1 person infected
        infected[time_start - 1] = 1 / item_population
        exposed[time_start - 1] = infected[time_start - 1] / alpha
        susceptible[time_start - 1] = 1 - exposed[time_start - 1] - infected[time_start - 1]

    # ----- NEW PARAMETER: What percent of the population remains susceptible? Take from recovered
    if percent_remaining_susceptible:
        for new_k in [k, k - 1]:
            s_diff = percent_remaining_susceptible - susceptible[new_k]
            susceptible[new_k] += s_diff
            recovered[new_k] -= s_diff
            susceptible[new_k] = min(1, susceptible[new_k])
            recovered[new_k] = max(0, recovered[new_k])

    # Update beta based on input Re
    r0 = rE / susceptible[k]
    beta = r0 / seir_params["length_infection"]

    # Run Deterministic Equations from Georgiy's PPT
    for k in range(time_start, time_start + time_limit):
        moving_out_s = beta * susceptible[k - 1] * infected[k - 1]
        moving_out_e = alpha * exposed[k - 1]
        moving_out_i = recovery_gamma * infected[k - 1]
        # Update vectors
        susceptible[k] = susceptible[k - 1] - moving_out_s
        exposed[k] = exposed[k - 1] + moving_out_s - moving_out_e
        infected[k] = infected[k - 1] + moving_out_e - moving_out_i
        recovered[k] = recovered[k - 1] + moving_out_i
        projection[k] = 1

    live_infections = np.round(infected * item_population, 0)
    new_infections = np.round(alpha * exposed * item_population, 0)
    # Replace with known values from df
    new_infections[0:time_start] = 0
    for day in df["Day"]:
        new_infections[day] = df.loc[df["Day"] == day, "Estimated Infections"].values[0]

    reported_cases = np.round(new_infections / case_multiplier, 0)
    cumulative_infections = new_infections.cumsum()

    out_df = pd.DataFrame()
    out_df["County"] = [item_name] * len(reported_cases)
    out_df["Day"] = [df.reset_index().Date[0] + dt.timedelta(i) for i in range(len(reported_cases))]
    out_df["cumulative_infections"] = cumulative_infections
    out_df["reported_cases"] = reported_cases
    out_df["expected_infections"] = new_infections
    out_df["live_infections"] = live_infections
    out_df["Projection"] = projection

    # if plot_data:
    #     plt1 = graphic(
    #         reported_cases,
    #         time_start,
    #         title="Reported Cases for " + item_name,
    #         ylabel="Reported Cases",
    #     )
    #     plt2 = graphic(
    #         cumulative_infections,
    #         time_start,
    #         title="Cumulative Infections for " + item_name,
    #         ylabel="Cumulative Infections",
    #     )
    #     return out_df, plt1, plt2

    return out_df
