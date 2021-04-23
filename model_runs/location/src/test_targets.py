from pathlib import Path

import pandas as pd
import plotly
import plotly.graph_objects as go

import src.data_input as di
from src.calibration_collection import format_daily_state
from src.constants import get_multiplier
from src.ldm import Ldm
from src.state import LifeState


def facility_population(ldm, locations: list = None):
    dc = ldm.daily_state.reset_index()
    dc = dc[dc.LIFE == LifeState.ALIVE.value]
    if locations:
        dc = dc[dc["LOCATION"].isin(locations)]
    return dc.drop(["LIFE", "LOCATION"], axis=1).sum()


def sum_movement(ldm, new: list = None, old: list = None):
    events = ldm.location_events
    if old:
        events = events[events["Location"].isin(old)]
    if new:
        events = events[events["New Location"].isin(new)]
    return pd.DataFrame(events.groupby(by="Time").size())


def steady_state_values(ldm, categories):
    steady_states = {}
    for category in categories:
        locations = ldm.nodes.category_ints[category]
        data = facility_population(ldm, locations)
        if data.shape[0] > 0:
            steady_states[category] = data[90:].mean()
        else:
            steady_states[category] = 0
    return steady_states


def equilibrium_matrix(ldm, categories: list):
    # How many people leave each node, and where do they go?
    rows = []
    for old_cat in categories:
        old_nodes = ldm.nodes.category_ints[old_cat]
        values = []
        for new_cat in categories:
            new_nodes = ldm.nodes.category_ints[new_cat]
            summed_events = sum_movement(ldm, new=new_nodes, old=old_nodes)
            value = summed_events.sum()[0]
            values.append(value)
        rows.append(values)

    df = pd.DataFrame(rows)
    df.index = categories
    df.columns = categories
    return df


def count_deaths(ldm, categories):
    le = ldm.life_events
    deaths = []
    for category in categories:
        locations = ldm.nodes.category_ints[category]
        deaths.append(int(le[le.Location.isin(locations)].shape[0]))

    return deaths


def make_targets(ldm):
    t = di.location_targets()

    # Targets: Section #1: Steady State for 5 Nodes
    population_multiplier = 1 / get_multiplier(ldm.params)
    categories = ["COMMUNITY", "UNC", "LARGE", "SMALL", "LT", "NH"]

    # The Values
    ss_values = steady_state_values(ldm, categories)
    ss_values.update((k, v * population_multiplier) for k, v in ss_values.items())

    # Targets: Section #2: 5x5 transition values
    df = equilibrium_matrix(ldm, categories).multiply(population_multiplier)
    # Deaths
    deaths = count_deaths(ldm, categories)
    deaths = [item * population_multiplier for item in deaths]

    df_lists = [list(item) for item in df.values]
    transition_values = [item for a_list in df_lists for item in a_list]
    transition_values = transition_values
    transition_values = [int(item) for item in transition_values]

    t["Values"] = list(ss_values.values())[1:] + transition_values + deaths[1:]

    t.to_csv(Path(ldm.run_dir, "model_output/targets.csv"), index=False)


if __name__ == "__main__":

    ldm = Ldm(scenario="location", run="run_calibration")
    ldm.run_model()
    ldm.life_events = ldm.life.events.make_events()
    ldm.location_events = ldm.movement.location.events.make_events()
    ldm.daily_state = format_daily_state(ldm)

    # The actual targets: This creates the targets.csv file for manual review
    make_targets(ldm)

    # what about ICU?
    print(pd.Series(ldm.icu_counts).mean())
    pd.Series(ldm.icu_counts).plot()

    # Check capacity at individual hospitals
    temp = ldm.daily_state.reset_index()
    temp = temp[temp.LIFE == 0].drop(["LIFE"], axis=1).set_index("LOCATION")
    temp = temp * (1 / get_multiplier(ldm.params))

    # What does ICU status look like over time:
    dc = ldm.daily_state.reset_index()
    dc = dc[dc.LIFE == LifeState.ALIVE.value]

    # Check Hospitals
    hospitals = di.hospitals().set_index("Name")

    fig = go.Figure()
    for h_int in ldm.nodes.category_ints["UNC"]:
        name = ldm.nodes.facilities[h_int].name
        values = temp.loc[h_int].values
        avg = values.mean()
        agents = hospitals.loc[name].Agents
        string = name + f". Goal: {round(agents, 0)}, Avg: {round(avg, 0)}"
        fig.add_trace(go.Scatter(x=list(range(0, 365)), y=temp.loc[h_int].values, name=string))

    plotly.offline.plot(fig, filename="file.html")
