from src.ldm import Ldm
import pandas as pd
import json
import tqdm
import random
import argparse as arg
from pathlib import Path
import os
from src.constants import POPULATION
import src.data_input as di
import numpy as np

if __name__ == "__main__":
    parser = arg.ArgumentParser()

    parser.add_argument(
        "--output_folder",
        default="../covid19-reporting/animated_visualization/data",
        help="The location to store the output data.",
    )

    parser.add_argument(
        "--county_fips", type=int, default=-1, help="A county fips code (not including state) to filter to.",
    )
    parser.add_argument(
        "--limit_pop", type=int, default=10000, help="The population size to run in the model.",
    )
    parser.add_argument(
        "--R0",
        type=float,
        default=1.2,
        help="The R0 value to use in this run (use to control the amount of covid in the visualization).",
    )

    parser.add_argument(
        "--ratio_to_hospital",
        type=float,
        default=0.0,
        help="The ratio of people with mild symptoms of covid who are hospitalized.",
    )
    parser.add_argument(
        "--community_probability_multiplier",
        type=float,
        default=0.79,
        help="A multiplier which controls the amount of people who go to hospitals.",
    )
    args = parser.parse_args()
    data_prop_dict = vars(args)

    num_days = 15

    output_dir = Path(args.output_folder)
    os.makedirs(output_dir, exist_ok=True)

    scenario = "../visualization"
    run = "parameters"

    base_params_file = Path("visualization", "parameters", "base_parameters.json")
    params_file = Path("visualization", "parameters", "parameters.json")

    r0 = args.R0

    with open(base_params_file, "r") as json_file:
        params = json.load(json_file)

    params["covid19"]["r0"] = r0
    params["covid19"]["r0_min"] = r0 - 0.2
    params["covid19"]["r0_max"] = r0 + 0.2
    params["covid19"]["ratio_to_hospital"] = args.ratio_to_hospital
    params["location"]["tuning"]["community_probability_multiplier"] = args.community_probability_multiplier
    params["base"]["limit_pop"] = args.limit_pop#max(args.limit_pop, 100000)

    with open(params_file, "w") as json_file:
        json.dump(params, json_file, indent=4)

    ldm = Ldm(scenario=scenario, run=run)

    life_list = {}
    covid_list = {}
    loc_list = {}
    cat_list = {}

    run_range = tqdm.trange(num_days, desc="---> COVID-Model:")
    for day in run_range:
        ldm.time = day
        ldm.step()
        life_list[day] = ldm.life.values.copy()
        covid_list[day] = ldm.disease.covid19.values.copy()
        loc_list[day] = ldm.movement.location.values.copy()
        if day == 0:
            cat_list[day] = [ldm.movement.facilities[x].category for x in loc_list[day]]
        else:
            cat_list[day] = cat_list[day - 1].copy()
            unique_ids = ldm.unique_ids[loc_list[day] != loc_list[day - 1]]
            for unique_id in unique_ids:
                cat_list[day][unique_id] = ldm.movement.facilities[loc_list[day][unique_id]].category

    loc_df = pd.DataFrame(loc_list)
    life_df = pd.DataFrame(life_list)
    covid_df = pd.DataFrame(covid_list)
    cat_df = pd.DataFrame(cat_list)

    file_type = ""

    if args.county_fips != -1:
        file_type += f"County_{args.county_fips}"
        # filter to the people in that county
        county_ids = ldm.unique_ids[ldm.county_codes == int(args.county_fips)]
        print("Filtering to ", len(county_ids), "people in county", args.county_fips)
        loc_df = loc_df.iloc[county_ids].copy()
        life_df = life_df.iloc[county_ids].copy()
        covid_df = covid_df.iloc[county_ids].copy()
        cat_df = cat_df.iloc[county_ids].copy()

    # count how many times someone changes locations
    changes = (loc_df[0] != 0).astype(int)
    for i in range(1, num_days):
        changes += (loc_df[i - 1] != loc_df[i]).astype(int)

    num_people = len(changes.index)
    # get everyone but order them by number of movements
    all_people = changes.sort_values(ascending=False).index.tolist()

    always_community = changes.loc[changes == 0].index.tolist()
    movers = changes.loc[changes > 0].index.tolist()

    # NOTE: not sure if this should be applied. If we sort by movements then
    # when the number of nodes is decreased there will still be movement
    print(f"Recording {num_people} actors for a model run with R0={r0}")

    # scale the bed counts by the subset of the population we are looking at
    actors_in_model = ldm.params["base"]["limit_pop"]
    bed_scale = actors_in_model / POPULATION
    data_prop_dict["Scale"] = POPULATION // actors_in_model

    facility_count_dict = {}
    bed_dict = {}

    nh_ids = di.nursing_homes()
    bed_dict["NH"] = nh_ids["Beds"].sum()
    facility_count_dict["NH"] = len(nh_ids)

    lt_ids = di.ltachs()
    bed_dict["LT"] = lt_ids["Beds"].sum()
    facility_count_dict["LT"] = len(lt_ids)

    loc_sizes = {"COMMUNITY": 0}
    cat_counts = {}
    for key, val in bed_dict.items():
        # at minimum, every facility has one bed
        loc_sizes[key] = max(max(round(val * bed_scale), 1), facility_count_dict[key])
        cat_counts[key] = 0

    # for hospitals, we just use the bed counts from the model itself
    for cat in ["SMALL", "LARGE", "UNC"]:
        loc_sizes[cat] = 0
        cat_counts[cat] = 0
    for f in ldm.nodes.all_hospitals:
        facility = ldm.nodes.facilities[f]
        loc_sizes[facility.category] += facility.n_total_beds

    # get the total number of actors who are ever in a cat during the model run
    # the final size is either the beds or this count, whichever is bigger
    all_cats = cat_df  # .loc[all_people]
    for col in all_cats.columns:
        cat_dict = all_cats[col].value_counts().to_dict()
        for cat in cat_counts:
            if cat in cat_dict.keys():
                cat_counts[cat] = max(cat_counts[cat], cat_dict[cat])

    for cat in cat_counts.keys():
        if cat_counts[cat] > loc_sizes[cat]:
            print(
                "Using", cat_counts[cat], "instead of the scaled beds", loc_sizes[cat], "for", cat,
            )
            loc_sizes[cat] = cat_counts[cat]

    # just assume nursing homes are about 70% full
    loc_sizes["NH"] = int(cat_counts["NH"] / 0.7)

    res_list = []
    for p in movers:

        res_list.append(
            {
                "ID": int(p),
                "covid_status": covid_df.loc[p, :].tolist(),
                "loc_cats": cat_df.loc[p, :].tolist(),
                "alive": life_df.loc[p, :].tolist(),
            }
        )

    community_list = []
    for p in always_community:

        community_list.append(
            {"ID": int(p), "covid_status": covid_df.loc[p, :].tolist(), "alive": life_df.loc[p, :].tolist(),}
        )

    res_dict = {"nodes": res_list, "loc_sizes": loc_sizes, "args": data_prop_dict, "community_nodes": community_list}

    with open(output_dir / f"vis_data.json", "w") as j_f:
        json.dump(res_dict, j_f, indent=4)
