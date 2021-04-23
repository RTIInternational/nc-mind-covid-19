from bisect import bisect
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def create_cdf(p: list) -> list:
    """ Convert a list of probabilities, into a cumulative distribution function

    Parameters
    ----------
    p : a list of probabilities that add to 1
    """
    s1 = sum(p)
    if (s1 < 0.999) or (s1 > 1.001):
        raise ValueError(f"You give a list, {p}, that does not sum to 1.")
    cdf = list()
    cdf.append(p[0])
    for i in range(1, len(p)):
        cdf.append(cdf[-1] + p[i])
    cdf[len(cdf) - 1] = 1
    return cdf


def normalize_and_create_cdf(p: list) -> list:
    """ Normalize a list of probabilities and create the cdf for them

    Parameters
    ----------
    p : list of probabilities. May not add to 1, as one probability may have been removed. Thus, we normalize
    """
    total = sum(p)
    if total == 0:
        return p
    p = [item / total for item in p]
    return create_cdf(p)


def random_selection(random: float, cdf: list, options: list) -> object:
    """ Given cumulative distribution function and a list of options, make a random selection

    Parameters
    ----------
    random: a random number between 0 and 1
    cdf : a list containing the cumulative distribution values. Ex [0, .2, .3, .7, 1.0]
    options : a list containing the options that can be selected
    """
    return options[bisect(cdf, random)]


def generate_distance_probability_distribution(county_df, distance_weight=1, bed_count_weight=1, closest_n=10):
    """ Given a county and it's distances to facilities as well as bed counts, calculate a
    probability that a person in that county will go to each facility.

    Parameters
    ----------
    county_df: a dataframe with the distance between each facility and the county
    cdf : a list containing the cumulative distribution values. Ex [0, .2, .3, .7, 1.0]
    options : a list containing the options that can be selected
    """
    if closest_n < len(county_df.index):
        county_df = county_df.sort_values(by="distance", ascending=True)[:closest_n]
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(county_df), columns=county_df.columns, index=county_df.index)

    scaled_df["dist_weighted"] = ((1 - scaled_df["distance"]) * distance_weight).apply(lambda x: max(x, 0.0001))
    scaled_df["bed_weighted"] = (scaled_df["beds"] * bed_count_weight).apply(lambda x: max(x, 0.0001))
    scaled_df["combined"] = scaled_df["dist_weighted"] * scaled_df["bed_weighted"]

    scaled_df["probability"] = scaled_df["combined"] / scaled_df["combined"].sum()

    return [create_cdf(scaled_df.probability.tolist()), scaled_df.index.tolist()]


def get_inverted_distance_probabilities(county_to_facility_distances):
    """ Given a dictionary of county to facility distances, invert it
    to get a probability distribution of each county given a facility based
    on distance
    Parameters
    ----------
    county_to_facility_distances: a dictionary of county IDs to lists of facilities
    and their distances to each county
    """
    facility_to_county_distances = {}
    for f in county_to_facility_distances["1"]:
        facility_to_county_distances[f["Name"]] = {}

    for county, distances in county_to_facility_distances.items():
        for f in distances:
            facility_to_county_distances[f["Name"]][int(county)] = f["distance_mi"]

    all_dist = pd.DataFrame(facility_to_county_distances)
    # we want closer distances to have higher probabilites so invert
    all_dist = 1 / all_dist
    # make these probability distributions
    prob_df = all_dist / all_dist.sum()

    prob_f_to_county_dict = {}
    for f in facility_to_county_distances.keys():
        # order by nearest county
        tmp_df = prob_df.sort_values(by=f, ascending=False)
        prob_f_to_county_dict[f] = [create_cdf(tmp_df[f].tolist()), tmp_df.index.tolist()]
    return prob_f_to_county_dict


sex_dictionary = {"M": 1, "F": 2}
race_dictionary = {"White": 1, "Black": 2, "Other": 3}
age_dictionary = {"L50": 0, "50-64": 1, "65+": 2}
