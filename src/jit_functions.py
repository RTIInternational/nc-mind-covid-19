from typing import Tuple

import numpy as np
from numba import njit


@njit
def assign_conditions(age: np.array, randoms: np.array):
    conditions = np.zeros(len(age), dtype=np.int8)
    for i in range(len(age)):
        if age[i] == 1:
            if randoms[i] < 0.2374:
                conditions[i] = 1
        elif age[i] == 2:
            if randoms[i] < 0.5497:
                conditions[i] = 1
    return conditions


@njit
def update_community_probability(cp: np.array, age: np.array, cc: np.array):
    """
    If simulating risk, we can update hospital transitions based on concurrent conditions. Update an agents
    community_probability based on their concurrent conditions and their age.
    - 55% age 1 should have concurrent conditions
    - 79% age 2 should have concurrent conditions
    """
    for i in range(len(age)):
        if cc[i] == 1:
            if age[i] == 1:
                cp[i] = cp[i] * 55 / 23.74
            elif age[i] == 2:
                cp[i] = cp[i] * 79 / 54.97
        else:
            if age[i] == 1:
                cp[i] = cp[i] * 45 / 76.26
            elif age[i] == 2:
                cp[i] = cp[i] * 21 / 45.03
    return cp


@njit("int64(boolean[:])")
def sum_true(array: np.array) -> int:
    return array.sum()


@njit("Tuple((boolean, int64))(boolean[:])")
def first_true_value(array: np.array) -> Tuple[bool, int]:
    """Finds the first True item in an array
    Args:
        array (np.array): Input array of boolean values
    Returns:
        Tuple[bool, int]: Tuple of (open_bed, bed_index)
    """
    for i, item in enumerate(array):
        if item is np.bool_(True):
            return (True, i)
    return (False, -1)


@njit
def init_daily_state(num_life_states: int, num_facilities: int, num_days: int) -> np.array:
    return np.zeros((num_life_states, num_facilities, num_days))


@njit
def init_beds_taken(num_facilities: int, num_icu_statuses: int, num_covid_states: int, num_days: int) -> np.array:
    return np.zeros((num_facilities, num_icu_statuses, num_covid_states, num_days))


@njit
def insert_daily_state(agent_life: np.array, agent_location: np.array, day: int, daily_state_data: np.array) -> None:
    """Iterates through the agents, incrementing the (life, location, day) value for each
    agent with that combination of attributes.

    Args:
        agent_life (np.array): Array of LifeState values
        agent_location (np.array): Array of Facility ID values
        day (int): The day / time of the simulation
        daily_state_data (np.array): The array of values. This function mutates this object and returns nothing.
    """
    for i in range(len(agent_life)):
        life = agent_life[i]
        location = agent_location[i]
        daily_state_data[life][location][day] += 1


@njit
def insert_beds_taken(
    agent_location: np.array, agent_icu: np.array, agent_covid: np.array, day: int, beds_taken_data: np.array
) -> None:
    """ Iterates through the agents, incrementing the (Location, ICU Status, COVID Status) value for each agent for the
    current day.

    Args:
        agent_location (np.array): Array of Facility ID values
        agent_icu (np.array): Array of ICU Statuses
        agent_covid (np.array): Array of COVID Statuses
        day (int): The current day of the simulation
        beds_taken_data (np.array): The array of values.
    """

    for i in range(len(agent_location)):
        location = agent_location[i]
        icu = agent_icu[i]
        covid = agent_covid[i]
        beds_taken_data[location][icu][covid][day] += 1


def map_array_from_dict(a, d):
    n = np.zeros(a.shape)
    for v in d:
        n[a == v] = d[v]
    return n
