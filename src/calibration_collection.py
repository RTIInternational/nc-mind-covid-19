import numpy as np
import pandas as pd

from src.jit_functions import init_daily_state, insert_daily_state, init_beds_taken, insert_beds_taken
from src.state import LifeState, IcuState, COVIDState


class Dynamic2DArray:
    """
    Expandable numpy array designed to be faster than np.append.
    Based on: https://stackoverflow.com/a/7134033

    Slightly slower than using Python lists but way more memory efficient.
    """

    __slots__ = ["num_columns", "capacity", "size", "data"]

    def __init__(self, num_columns: int, dtype=np.uint32):
        self.num_columns = num_columns
        self.capacity = 100
        self.size = 0
        self.data = np.zeros((self.capacity, self.num_columns), dtype=dtype)

    def add_row(self, row: np.ndarray):
        if self.size == self.capacity:
            self.capacity *= 2
            newdata = np.zeros((self.capacity, self.num_columns))
            newdata[: self.size] = self.data
            self.data = newdata

        self.data[self.size] = row
        self.size += 1

    def finalize(self):
        return self.data[: self.size]

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """
        Note: allowing access to this might allow users to do something unexpected
        if they set rows that don't "exist" yet but are part of the capacity.
        """
        return self.data.__setitem__(*args, **kwargs)


class EventStorage:
    __slots__ = ["column_names", "column_indices", "arr", "tracking"]

    def __init__(self, column_names: list, tracking: bool = True):
        self.column_names = column_names
        self.column_indices = {c: i for i, c in enumerate(self.column_names)}
        self.arr = Dynamic2DArray(len(self.column_names))
        self.tracking = tracking

    def record_state_change(self, row: tuple):
        if self.tracking:
            self.arr.add_row(np.array(row))

    def record_event(self, row: tuple):
        self.arr.add_row(np.array(row))

    def make_events(self) -> pd.DataFrame:
        return pd.DataFrame(self.arr.finalize(), columns=self.column_names)

    def __repr__(self):
        cn = "Columns for this event: {}".format(self.column_names)
        return "EventStorage: tracking is current {}. {}".format("ON" if self.tracking else "OFF", cn)


def init_daily_state_data(ldm) -> np.array:
    return init_daily_state(
        num_life_states=len(LifeState),
        num_facilities=len(ldm.facility_ids),
        num_days=ldm.params["base"]["time_horizon"],
    )


def init_beds_taken_data(ldm) -> np.array:
    return init_beds_taken(
        num_facilities=len(ldm.facility_ids),
        num_icu_statuses=2,
        num_covid_states=len(COVIDState),
        num_days=ldm.params["base"]["time_horizon"],
    )


def step_daily_state_data(ldm) -> None:
    if ldm.track_events:
        insert_daily_state(ldm.life.values, ldm.movement.location.values, ldm.time, ldm.daily_state_data)


def step_beds_taken_data(ldm) -> None:
    if ldm.run_covid:
        unique_ids = ldm.unique_ids[ldm.movement._agents_in_hospitals]
        insert_beds_taken(
            ldm.movement.location.values[unique_ids],
            ldm.icu_status[unique_ids],
            ldm.disease.covid19.values[unique_ids],
            ldm.time,
            ldm.beds_taken_data,
        )


def format_daily_state(ldm) -> pd.DataFrame:
    """Formats the daily counts array into a DataFrame

    Returns:
        pd.DataFrame: DataFrame indexed by (LIFE, LOCATION), columns=days.
    """
    daily_state_numba_ndx = pd.MultiIndex.from_product(
        [[e.value for e in LifeState], ldm.facility_ids, list(range(ldm.params["base"]["time_horizon"]))],
        names=["LIFE", "LOCATION", "day"],
    )
    daily_state = pd.DataFrame(ldm.daily_state_data.flatten(), index=daily_state_numba_ndx).unstack()
    daily_state.columns = list(range(ldm.params["base"]["time_horizon"]))
    daily_state = daily_state.sort_index()
    return daily_state


def format_beds_taken(ldm) -> pd.DataFrame:
    """Formats the daily beds taken array into a DataFrame

    Returns:
        pd.DataFrame: DataFrame indexed by (LIFE, LOCATION), columns=days.
    """
    beds_taken_numba_ndx = pd.MultiIndex.from_product(
        [
            ldm.facility_ids,
            [icu.value for icu in IcuState],
            [c.value for c in COVIDState],
            list(range(ldm.beds_taken_data.shape[3])),
        ],
        names=["LOCATION", "ICU_STATUS", "COVID_STATUS", "day"],
    )
    beds_taken = pd.DataFrame(ldm.beds_taken_data.flatten(), index=beds_taken_numba_ndx).unstack()
    beds_taken.columns = list(range(ldm.beds_taken_data.shape[3]))
    beds_taken = beds_taken.sort_index()
    return beds_taken
