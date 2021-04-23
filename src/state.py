from enum import IntEnum

import numpy as np

from src.jit_functions import update_community_probability


class LifeState(IntEnum):
    ALIVE = 0
    DEAD = 1


class IcuState(IntEnum):
    NO = 0
    YES = 1


class COVIDState(IntEnum):
    SUSCEPTIBLE = 0
    COVID19MILD = 1
    COVID19SEVERE = 2
    COVID19CRITICAL = 3
    RECOVERED = 4
    DEAD = 5


class COVIDTest(IntEnum):
    NA = 0
    TESTED = 1
    UNTESTED = 2


class AgeGroup(IntEnum):
    AGE0 = 0
    AGE1 = 1
    AGE2 = 2


class ConcurrentConditions(IntEnum):
    NO = 0
    YES = 1


class NcCategories(IntEnum):
    COMMUNITY = 0
    UNC = 1
    LARGE = 2
    SMALL = 3
    LT = 4
    NH = 5


LifeState.id = "LIFE"
AgeGroup.id = "AGE"
NcCategories.id = "NcCategories"


class EventState:
    __slots__ = [
        "enum",
        "integers",
        "transition_dict",
        "key_types",
        "values",
        "probabilities",
        "events",
        "previous",
        "integers",
        "names",
    ]

    def __init__(self, enum: IntEnum, transition_dict: dict, key_types: list):
        self.enum = enum
        if self.enum:
            self.integers = [item.value for item in enum]
            self.names = [item.name for item in enum]
        self.transition_dict = transition_dict
        self.key_types = key_types
        self.values = None
        self.probabilities = None
        self.events = None
        self.previous = None

    def initiate_values(self, count: int, value: int, dtype: type = int) -> np.array:
        self.values = np.zeros(count, dtype=dtype)
        self.values.fill(value)

    def update_community_probability(self, cp, age, cc):
        self.probabilities = update_community_probability(cp=cp, age=age, cc=cc)

    def find_probabilities(self, keys: list):
        probabilities = np.zeros(len(keys))
        for i in range(len(keys)):
            probabilities[i] = self.transition_dict[keys[i]]
        return probabilities

    def copy_values_to_previous(self):
        self.previous = self.values.copy()

    def __repr__(self):
        return "Values and transition probabilities for: {}".format(self.enum)


class CovidStateContainer:
    __slots__ = ["values", "states"]


class Empty:
    """ An empty state to house extra arrays or dictionaries
    """

    def __init__(self, data_type):
        self.data_type = data_type
