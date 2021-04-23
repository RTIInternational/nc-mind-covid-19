from typing import Union

import numpy as np

from src.data_processing import county_name_to_flu_region_map
from src.jit_functions import first_true_value, sum_true


class Hospital:
    __slots__ = [
        "name",
        "category",
        "base_n_total_beds",
        "base_n_icu_beds",
        "base_n_ventilator_beds",
        "base_n_normal_beds",
        "initial_normal",
        "initial_icu",
        "initial_normal_covid",
        "initial_icu_covid",
        "county",
        "region_str",
        "agents",
        "normal_beds",
        "icu_beds",
        "ventilator_beds",
        "n_total_beds",
        "n_icu_beds",
        "n_ventilator_beds",
        "n_normal_beds",
        "model_int",
        # Called by `north_carolina.py`
        "options",
        "los_for_sampling",
    ]

    def __init__(
        self,
        name: str,
        county: str,
        h_category: str,
        n_beds: int,
        n_icu_beds: int,
        n_ventilator_beds: int,
        initial_normal: float,
        initial_icu: float,
        initial_normal_covid: int,
        initial_icu_covid: int,
    ):

        if n_icu_beds < n_ventilator_beds:
            raise ValueError("Number of Ventilators must be less than or equal to the Number of ICU Beds")

        self.name = name
        self.category = h_category

        self.base_n_total_beds = self.n_total_beds = n_beds
        self.base_n_icu_beds = self.n_icu_beds = n_icu_beds
        self.base_n_ventilator_beds = self.n_ventilator_beds = n_ventilator_beds
        self.base_n_normal_beds = self.n_normal_beds = n_beds - n_icu_beds

        self.initial_normal = initial_normal
        self.initial_icu = initial_icu
        self.initial_normal_covid = initial_normal_covid
        self.initial_icu_covid = initial_icu_covid

        self._create_bed_arrays(self.n_total_beds, self.n_icu_beds, self.n_ventilator_beds)

        self.county = county
        self.region_str = county_name_to_flu_region_map().get(county)

        # called in north_carolina.py
        self.los_for_sampling = None

    def _create_bed_arrays(self, n_total_beds: int, n_icu_beds: int, n_ventilator_beds: int):
        self.agents = np.full(shape=n_total_beds, fill_value=-1, dtype=np.int64)
        self.normal_beds = _create_partial_boolean_array(length=n_total_beds, n_filled=n_icu_beds, invert=True)
        self.icu_beds = _create_partial_boolean_array(length=n_total_beds, n_filled=n_icu_beds)
        self.ventilator_beds = _create_partial_boolean_array(length=n_total_beds, n_filled=n_ventilator_beds)

    def apply_bed_multiplier(self, multiplier: float):
        self.n_total_beds = max(1, round_to_int(self.base_n_total_beds * multiplier))
        self.n_icu_beds = round_to_int(self.base_n_icu_beds * multiplier + 0.1)
        self.n_ventilator_beds = round_to_int(0.95 * self.n_icu_beds)
        self.n_normal_beds = self.n_total_beds - self.n_icu_beds

        self._create_bed_arrays(self.n_total_beds, self.n_icu_beds, self.n_ventilator_beds)

    def empty_beds(self, icu: bool = False, ventilator: bool = False, normal: bool = False) -> np.array:
        """Find empty beds meeting certain criteria
        Args:
            icu (bool, optional): Beds need to be ICU. Defaults to False.
            ventilator (bool, optional): Beds need to include Ventilator. Defaults to False.
            normal (bool, optional): Normal bed availability. Defaults to False.
        Returns:
            np.array: array of `bool` values indicating bed availability by index
        """

        if normal:
            return self.open_normal_beds
        if ventilator:
            return self.open_ventilator_beds
        if icu:
            return self.open_icu_beds
        return self.open_beds

    def occupied_beds(self, icu: bool = False, ventilator: bool = False, normal: bool = True):
        if normal:
            return ~self.open_normal_beds & self.normal_beds
        if ventilator:
            return ~self.open_ventilator_beds & self.ventilator_beds
        if icu:
            return ~self.open_icu_beds & self.icu_beds
        return ~self.open_beds

    def add_agent(self, unique_id: int, icu: bool = False, ventilator: bool = False, normal: bool = True) -> bool:
        """Adds an agent to an empty bed in this Hospital (if available)
        Args:
            unique_id (int): Agent ID
            icu (bool, optional): Agent needs an ICU Bed. Defaults to False.
            ventilator (bool, optional): Agent needs a Ventilator. Defaults to False.
            normal (bool, optional): Agent needs a normal bed. Defaults to True.
        Returns:
            bool: True if agent was added, False if no open beds
        """
        if icu | ventilator:
            normal = False
        open_bed, bed_id = first_true_value(self.empty_beds(icu=icu, ventilator=ventilator, normal=normal))
        if open_bed:
            self.agents[bed_id] = unique_id
        return open_bed

    def remove_agent(self, unique_id: int) -> None:
        """Removes an agent from the hospital by setting the value
        in the agents array to -1
        Args:
            unique_id (int): Unique Agent ID
        """
        self.agents[self.agents == unique_id] = -1

    @property
    def open_beds(self) -> np.array:
        return self.agents < 0

    @property
    def open_normal_beds(self) -> np.array:
        return self.open_beds & self.normal_beds

    @property
    def open_icu_beds(self) -> np.array:
        return self.open_beds & self.icu_beds

    @property
    def open_ventilator_beds(self) -> np.array:
        return self.open_beds & self.ventilator_beds

    def capacity(
        self, icu: bool = False, ventilator: bool = False, normal: bool = False, count: bool = False
    ) -> Union[float, int]:
        """Calculates percent full (capacity)
        Args:
            icu (bool, optional): Capacity of ICU beds only. Defaults to False.
            ventilator (bool, optional): Capacity of Ventilator beds only. Defaults to False.
            normal (bool, optional): normal capacity . Defaults to False.
            count (bool, optional): Return Count, rather than percent. Defaults to False.
        Returns:
            Union[float, int]: Capacity (percent or count occupied)
        """

        num = sum_true(self.occupied_beds(icu=icu, ventilator=ventilator, normal=normal))
        den = num + sum_true(self.empty_beds(icu=icu, ventilator=ventilator, normal=normal))

        if count:
            return num
        else:
            if den == 0:
                return 0
            return num / den

    def capacity_given_array(
        self,
        array: np.array,
        value: int,
        icu: bool = False,
        ventilator: bool = False,
        normal: bool = False,
        count: bool = False,
    ) -> Union[float, int]:
        """Calculates percent full (capacity)
        Args:
            array (array, required): an array of values corresponding to the agents
            value (int, required): the int value to filter the array to
            icu (bool, optional): Capacity of ICU beds only. Defaults to False.
            ventilator (bool, optional): Capacity of Ventilator beds only. Defaults to False.
            normal (bool, optional): normal capacity . Defaults to False.
            count (bool, optional): Return Count, rather than percent. Defaults to False.
        Returns:
            Union[float, int]: Capacity (percent or count occupied)
        """
        beds = self.occupied_beds(icu=icu, ventilator=ventilator, normal=normal)
        ids = np.where(beds)[0]
        people = [b for b in ids if array[self.agents[b]] == value]
        if count:
            return len(people)

        num = sum_true(self.occupied_beds(icu=icu, ventilator=ventilator, normal=normal))
        den = num + sum_true(self.empty_beds(icu=icu, ventilator=ventilator, normal=normal))
        if den == 0:
            den = 1

        return len(people) / den


class BaseFacility:
    __slots__ = ["agents", "options"]

    def __init__(self):
        self.agents = set()

    def add_agent(self, unique_id):
        self.agents.add(unique_id)
        return True

    def remove_agent(self, unique_id):
        self.agents.discard(unique_id)


class NursingHome(BaseFacility):
    __slots__ = ["beds", "name", "category", "model_int"]

    def __init__(self, beds: int, name: str, model_int: int):
        super().__init__()
        self.beds = beds
        self.name = name
        self.category = "NH"
        self.model_int = model_int


class LongTermCareFacility(BaseFacility):
    __slots__ = ["beds", "name", "category", "model_int"]

    def __init__(self, beds: int, name: str, model_int: int):
        super().__init__()
        self.beds = beds
        self.name = name
        self.category = "LT"
        self.model_int = model_int


class Community(BaseFacility):
    __slots__ = ["name", "category", "model_int"]

    def __init__(self):
        super().__init__()
        self.name = "COMMUNITY"
        self.category = "COMMUNITY"


def _create_partial_boolean_array(length: int, n_filled: int, invert=False):
    array = np.full(shape=length, fill_value=False)
    array[:n_filled] = True
    return ~array if invert else array


def round_to_int(x: float) -> int:
    return int(round(x))
