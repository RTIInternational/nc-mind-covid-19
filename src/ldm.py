from pathlib import Path
import random

import numpy as np
import tqdm

from src.constants import get_multiplier
from src.covid import COVIDModel
from src.data_processing import county_to_flu_region_map, flu_region_map, sample_population
from src.jit_functions import assign_conditions, map_array_from_dict
from src.life import Life
from src.north_carolina import NorthCarolina
from src.north_carolina_nodes import NcNodeCollection
from src.parameters import load_parameters
from src.state import AgeGroup, LifeState, NcCategories, IcuState
from src.calibration_collection import (
    init_daily_state_data,
    init_beds_taken_data,
    step_daily_state_data,
    step_beds_taken_data,
)


class Ldm:
    def __init__(self, scenario: str, run: str):
        """Ldm: Location & Disease model - a class built to run agent-based simulations."""

        # Setup the model directory structure
        if "tests" in scenario:
            self.scenario_dir = Path(scenario)
        else:
            self.scenario_dir = Path("model_runs", scenario)
        self.run_dir = Path(self.scenario_dir, run)
        self.output_dir = Path(self.run_dir, "model_output")
        self.run_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # self.track_events = scenario == "location" and run == "run_calibration"
        self.track_events = True

        # Setup the model parameters
        self.params = load_parameters(Path(self.run_dir, "parameters.json"))
        self.seed = self.params["base"]["seed"]
        self.time_horizon = self.params["base"]["time_horizon"]
        self.rng = np.random.RandomState(self.seed)
        self.rng_stdlib = random.Random(self.seed)

        self.time = 0

        self.unique_regions = {"state": 1}

        # Population
        self.population = sample_population(self.params["base"]["limit_pop"], self.seed)
        self.unique_ids = np.array(self.population.index.values, dtype=np.int32)
        self.icu_status = np.zeros(self.population.shape[0], dtype=np.int8)
        self.icu_counts = []
        self.icu_status.fill(IcuState.NO.value)
        self.county_codes = self.population.County_Code.values
        self._unique_county_codes = np.unique(self.county_codes)
        self._county_code_mask = {code: self.county_codes == code for code in self._unique_county_codes}
        self.map_regions()
        self.age_groups = self.population.Age_Group.values
        self.ages = self.population.Age_Years.values
        self.population = self.population.drop(["County_Code", "Age_Group", "Age_Years", "logrecno"], axis=1)

        self.concurrent_conditions = assign_conditions(self.age_groups, self.rng.rand(len(self.population)))
        self.nodes = NcNodeCollection(get_multiplier(self.params))
        self.movement = NorthCarolina(model=self)
        self.run_covid = "covid19" in self.params.keys()
        self.disease = None
        if self.run_covid:
            self.disease = COVIDModel(model=self, params=self.params["covid19"])

        life_dict = dict()
        for k1, v1 in self.params["life"]["death_probabilities"].items():
            for k2, v2 in self.params["life"]["death_multipliers"].items():
                life_dict[AgeGroup[k1].value, NcCategories[k2].name] = v1 * v2
        self.life = Life(model=self, enum=LifeState, transition_dict=life_dict, key_types=[AgeGroup, NcCategories])
        # Death probability is based on Age + Location
        current_locations = [self.nodes.facilities[item].category for item in self.movement.location.values]
        self.life.probabilities = self.life.find_probabilities(list(zip(self.age_groups, current_locations)))

        self.facility_ids = list(self.nodes.facilities.keys())

        self.daily_state_data = init_daily_state_data(self)
        self.beds_taken_data = init_beds_taken_data(self)

    def run_model(self):
        for day in tqdm.trange(0, self.time_horizon, desc="---> Model is Running"):
            self.time = day
            self.step()

    def step(self):
        self.life.step()
        self.movement.step()
        step_daily_state_data(self)
        self.regenerate_agents()
        step_beds_taken_data(self)
        self.icu_counts.append(self.icu_status.sum())

    def regenerate_agents(self):
        """Every 15 days, regenerate new agents for the agents who have died"""
        agent_ids = []
        if (self.time % 15 == 0) and (len(self.life.agents_to_recreate) > 0):
            agent_ids = [item for item in self.life.agents_to_recreate if item < self.population.shape[0]]
        if len(agent_ids) > 0:
            l1 = len(self.unique_ids)
            self.unique_ids = np.append(self.unique_ids, [range(l1, l1 + len(agent_ids))])
            self.county_codes = np.append(self.county_codes, self.county_codes[agent_ids])
            self._county_code_mask = {code: self.county_codes == code for code in self._unique_county_codes}
            self.regions = np.append(self.regions, self.regions[agent_ids])
            self.age_groups = np.append(self.age_groups, self.age_groups[agent_ids])
            self.icu_status = np.append(self.icu_status, np.array([IcuState.NO.value] * len(agent_ids), dtype=np.int8))
            self.ages = np.append(self.ages, self.ages[agent_ids])

            # Assign concurrent conditions
            conditions = assign_conditions(self.age_groups[agent_ids], self.rng.rand(len(agent_ids)))
            self.concurrent_conditions = np.append(self.concurrent_conditions, conditions).astype(np.int8)

            self.movement.regenerate_agents(agent_ids)
            self.disease.regenerate_agents(agent_ids) if self.run_covid else None
            self.life.agents_to_recreate = []

    def map_regions(self):
        self.unique_regions = flu_region_map()
        unique_county_map = {county: self.unique_regions[region] for county, region in county_to_flu_region_map(

        ).items()}

        self.regions = map_array_from_dict(self.county_codes, unique_county_map)
