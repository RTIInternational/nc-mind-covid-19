import math
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as stats

import src.data_input as di
from seir.src.run_seir import run_seir
from src.calibration_collection import EventStorage
from src.constants import POPULATION
from src.state import AgeGroup, COVIDState, COVIDTest, EventState, CovidStateContainer
from src.data_processing import county_to_coalition_map
from src.jit_functions import map_array_from_dict


class COVIDModel:
    def __init__(self, model, params):
        """COVID Model #1
        This model hopes to assess how quickly hospitals will fill up if cases of COVID-19 follow similar patterns in NC
        as they have in other locations. The parameters fill will allow users to assess different trajectories
        """
        self.model = model
        self.params = params
        self.cases = run_seir(
            rE=params["r0"],
            use_correction=True,
            case_multiplier=params["initial_case_multiplier"],
            percent_remaining_susceptible=params["percent_remaining_susceptible"],
            time_limit=self.model.time_horizon + 1,
        )

        self.start_date = pd.Timestamp(self.cases[self.cases.Projection == 0].Day.max().date()) + pd.Timedelta(1, "d")

        self.covid19 = CovidStateContainer()
        self.covid19.values = np.zeros(len(self.model.population), dtype=np.int16)
        self.covid19.values.fill(COVIDState.SUSCEPTIBLE.value)
        self.covid19.states = [item.name for item in COVIDState]

        # Probability of testing is the same for all age groups
        test_dict = {0: self.params["p_tested"], 1: self.params["p_tested"], 2: self.params["p_tested"]}
        self.covid19tested = EventState(enum=COVIDTest, transition_dict=test_dict, key_types=[AgeGroup])
        self.covid19tested.initiate_values(count=len(self.model.population), value=COVIDTest.NA.value)
        self.covid19tested.probabilities = self.covid19tested.find_probabilities(self.model.age_groups)

        self.recovery_day = dict()

        # ----- Case counts by day by case type
        self.covid_cases = EventStorage(
            column_names=["Unique_ID", "Time", "Type", "ICU", "Ventilator", "Seeking Hospital"]
        )
        self.nh_demand_by_day = dict()

        for i in self.model.unique_regions.values():
            self.nh_demand_by_day[i] = {"daily_patients": defaultdict(int)}

        # Sum all COVID19 cases
        # self.sum_initial_cases = 0
        self.initial_cumulative_infections = 0
        self.initial_live_infections = 0

        # County Parameters
        self.county_info = defaultdict(dict)
        self.counties = self.params["counties"]
        if len(self.counties) == 0:
            self.counties = [i for i in range(1, 201, 2)]
        self.fill_county_info()

        # Census of COVID patients in a hospital bed
        self.covid_beds_by_day = dict()
        for i in self.model.unique_regions.values():
            self.covid_beds_by_day[i] = {
                "hospital_census_acute_bed": defaultdict(int),
                "hospital_census_icu_bed": defaultdict(int),
            }

        # Create weights to assign covid by age
        (
            self.covidassignment_age_dict,
            self.covidassignment_weights_dict,
            self.covid_weights,
        ) = self.find_covid_weights_by_age(
            self.params["distributions"]["ages_getcovid"]["ranges"],
            self.params["distributions"]["ages_getcovid"]["distribution"],
        )
        self.events = EventStorage(column_names=["Unique_ID", "Time", "COVID_State"], tracking=False)
        self._init_p_hospitalized()
        self._init_live_infection = self.model.rng.rand(len(self.covid19.values)) < self.proportion_initial_live
        self._sample_first_los = self.init_los_from_truncnorm()

        # get the number of initial covid hospitalizations needed
        self._initial_hosp_covid_needed = {}
        total_hosp = 0
        for i in self.model.nodes.all_hospitals:
            facility = self.model.nodes.facilities[i]
            # need to adjust the covid bed counts to the number of beds actually created in the facilities
            if facility.base_n_normal_beds == 0:
                acute = 0
            else:
                acute = facility.n_normal_beds * (facility.initial_normal_covid / facility.base_n_normal_beds)
                acute = int(acute + self.model.rng.rand())

            if facility.base_n_icu_beds == 0:
                icu = 0
            else:
                icu = facility.n_icu_beds * (facility.initial_icu_covid / facility.base_n_icu_beds)
                icu = int(icu + self.model.rng.rand())

            # adjust for the non-covid people already added
            noncovid_acute_beds = int(
                facility.n_normal_beds / max(facility.base_n_normal_beds, 1) * facility.initial_normal
            )
            noncovid_icu_beds = int(facility.n_icu_beds / max(facility.base_n_icu_beds, 1) * facility.initial_icu)

            acute = min(facility.n_normal_beds - noncovid_acute_beds, acute)
            icu = min(facility.n_icu_beds - noncovid_icu_beds, icu)

            total_hosp += icu + acute
            self._initial_hosp_covid_needed[facility.name] = {"acute": acute, "icu": icu}

        # this controls the probability that we will force a covid infected person into a hospital
        # The idea is that we don't want hospitals being filled up by one county
        self._initial_hosp_prob = min(0.9, 0.1 + (total_hosp / self.initial_live_infections))

        # get list of closest hospitals for each county (to use later)
        dist = di.county_hospital_distances()
        self._county_close_hosp = {}
        for c in dist.keys():
            self._county_close_hosp[c] = []
            for h in dist[c]:
                if h["distance_mi"] >= 60:
                    break
                self._county_close_hosp[c].append(h["Name"])

    def init_los_from_truncnorm(self):
        """Initialize some length of stays to sample from for a unc hospital
        according to the following steps:
        1. Initialize a large number of los from the truncated normal distribution
        2. Calculate how many days it would take to stabalize the los by getting
        the 95th percentile of the distribution
        3. Step the LOS forward that many days. When a LOS reaches 0, sample from the
        distribution for a new number to replace it
        """

        lower, upper = self.params["los_min"], self.params["los_max"]
        mu, sigma = self.params["los_mean"], self.params["los_std"]
        initial_sample = stats.truncnorm.rvs(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=100000
        )
        initial_sample = np.rint(initial_sample)

        num_days = int(np.percentile(initial_sample, 95))

        sample = pd.Series(initial_sample)
        for day in range(num_days):
            sample -= 1
            a_filter = sample < 0
            sample[a_filter] = self.model.rng_stdlib.choices(initial_sample, k=sample[a_filter].size)
        return sample.tolist()

    def _init_p_hospitalized(self):
        tested_concurrent = (
            (COVIDTest.TESTED.value, 0),
            (COVIDTest.TESTED.value, 1),
            (COVIDTest.UNTESTED.value, 0),
            (COVIDTest.UNTESTED.value, 1),
        )
        params_lookup = (
            "p_covidhospitalized_tested_noconcurrent",
            "p_covidhospitalized_tested_concurrent",
            "p_covidhospitalized_nottested_noconcurrent",
            "p_covidhospitalized_nottested_concurrent",
        )
        hosp_lookup = dict(zip(tested_concurrent, params_lookup))
        self.p_hospitalized_lookup = {}
        for (tested, concurrent) in tested_concurrent:
            age_distribution = self.params["distributions"][hosp_lookup[(tested, concurrent)]]["distribution"]
            for age_index in range(len(age_distribution)):
                self.p_hospitalized_lookup[tested, concurrent, age_index] = age_distribution[age_index]

    def seir_covid19_cases(self) -> pd.DataFrame:
        df = self.cases
        county_code_df = di.county_codes()
        df = df.merge(county_code_df[["County", "County_Code"]], how="left", left_on="County", right_on="County")
        coalition_map = county_to_coalition_map()
        df["coalition"] = df["County_Code"].apply(lambda x: coalition_map[x] if x in coalition_map.keys() else None)

        return df

    def fill_county_info(self):
        df = self.seir_covid19_cases()

        temp_df = df[(df.Day == pd.Timestamp(self.start_date)) & (df.County == "North Carolina")]
        self.initial_cumulative_infections = temp_df["cumulative_infections"].values[0]
        self.initial_live_infections = temp_df["live_infections"].values[0]
        self.proportion_initial_live = self.initial_live_infections / self.initial_cumulative_infections

        # Remove State and Regions
        df = df[(df["County"] != "North Carolina") & ([isinstance(i, str) for i in df["County"]])]

        init_df = df[(df.Day == pd.Timestamp(self.start_date))]
        self.initial_coalition_infections = init_df.groupby("coalition")["live_infections"].sum().to_dict()

        for _, row in df[df.Day >= pd.Timestamp(self.start_date)].iterrows():
            self.county_info[row.Day][row.County_Code] = {
                "cumulative_infections": row["cumulative_infections"],
                "expected_infections": row["expected_infections"],
            }

    def go_to_hospital(
        self, unique_id: int, new_location: int, covid_state: int, icu: bool = False, ventilator: bool = False
    ):
        """
        Attempt to send a COVID agent to a hospital using the same process as for non-COVID agents.
        If agent is turned away, count them here to get a count of COVID agents specifically turned away.
        """

        self.model.movement.move_to_stach(
            unique_id=unique_id,
            current_location=0,
            new_location=new_location,
            force_icu=icu,
            test_icu=False,
            ventilator=ventilator,
            covid_state=covid_state,
        )

        # If patient was turned away: Do nothing - they did not make it to a hospital
        if self.model.movement.location.values[unique_id] == 0:
            self.recovery_day[unique_id] = self.model.time + self.params["infection_duration"]
            return
        # Overwrite their LOS
        lower, upper = self.params["los_min"], self.params["los_max"]
        mu, sigma = self.params["los_mean"], self.params["los_std"]
        if self._first_date:
            los = self.model.rng.choice(self._sample_first_los)
            los = int(round(los))
            los = max(los, 0)
        else:
            los_continuous = stats.truncnorm.rvs(
                (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=1
            )[0]
            los = int(round(los_continuous, 0))
            los = max(los, 1)
        self.model.movement.assign_los(unique_id=unique_id, new_location=new_location, los=los)
        self.recovery_day[unique_id] = self.model.time + los

        self.covid_cases.record_event((unique_id, self.model.time, covid_state, int(icu), int(ventilator), int(True)))

    def step(self):
        self._first_date = self.model.time == 0
        self._succeptible_pop = self.covid19.values == COVIDState.SUSCEPTIBLE
        self._community_pop = self.model.movement.location.values == 0
        self._succeptible_community_pop = self._succeptible_pop & self._community_pop
        self.model.rng.shuffle(self.counties)
        for county in self.counties:
            x = self.find_new_cases(county)
            self.make_cases(county, x)

    def find_new_cases(self, county_code: int):
        """Each day, new cases of covid19 arise in the community. This is based on input parameters
        and implemented in each county.
        """
        scaler = self.model.population.shape[0] / POPULATION

        # on day 0, all cumulative cases up to that day, plus the expected cases for that day,
        # should be put given infections
        if self._first_date:
            initial = self.county_info[self.start_date][county_code]["cumulative_infections"]
            initial += self.county_info[self.start_date][county_code]["expected_infections"]
            expected_cases = max(1, initial)
            expected_cases = expected_cases * scaler
        # after day 0, just use the expected cases
        else:
            date = self.start_date + pd.Timedelta(self.model.time, "d")
            expected_cases = self.county_info[date][county_code]["expected_infections"] * scaler

        new_cases = math.floor(expected_cases)
        if self.model.rng.rand() < (expected_cases - new_cases):
            new_cases += 1
        if new_cases < 0:
            new_cases = 0

        return new_cases

    def make_cases(self, county_code, number_of_cases):
        """Find agents in the community and give them COVID19 - Scaled based on population"""
        # Must be susceptible and in the community
        unique_ids = self.model.unique_ids[
            (self.model._county_code_mask[county_code]) & self._succeptible_community_pop
        ]
        # Must be enough people
        number_of_cases = min(len(unique_ids), number_of_cases)

        if len(unique_ids) > 0:
            n_cases = int(round(number_of_cases))
            weights = self.covid_weights[unique_ids]
            weights = weights / weights.sum()
            unique_ids = self.model.rng.choice(unique_ids, size=n_cases, replace=False, p=weights)
            self.model.rng.shuffle(unique_ids)
            for unique_id in unique_ids:
                self.give_covid19(unique_id, county_code)

    def is_tested(self, unique_id: int) -> int:
        """Determine if agent is tested for COVID19"""
        if self.model.rng.rand() < self.covid19tested.probabilities[unique_id]:
            return COVIDTest.TESTED.value
        return COVIDTest.UNTESTED.value

    def give_covid19(self, unique_id: int, county_code: int):
        """Give an agent COVID19 with symptom severity determined by probability"""

        hospital_int = None
        # If it is the first date, determine whether the case is live
        if self._first_date:

            if self._init_live_infection[unique_id]:
                if self.model.rng.rand() < self._initial_hosp_prob:
                    # try to fit them in a nearby hospital
                    for h in self._county_close_hosp[county_code]:
                        # skip hospitals which are in distance file but not in the model
                        if h not in self.model.nodes.name_to_int.keys():
                            continue
                        if self._initial_hosp_covid_needed[h]["acute"] > 0:
                            # send them to this hospital in acute bed
                            p_acute = 1
                            p_icu = 0
                            self._initial_hosp_covid_needed[h]["acute"] -= 1
                            hospital_int = self.model.nodes.name_to_int[h]
                            break
                        elif self._initial_hosp_covid_needed[h]["icu"] > 0:
                            # send them to this hospital in icu bed
                            p_acute = 0
                            p_icu = 1
                            self._initial_hosp_covid_needed[h]["icu"] -= 1
                            hospital_int = self.model.nodes.name_to_int[h]
                            break
                    # no hospitals nearby needed people, keep them in the community
                    if hospital_int is None:
                        p_acute = 0
                        p_icu = 0

                else:
                    # they are not hospitalized
                    p_acute = 0
                    p_icu = 0

            # agent is recovered and is not hospitalized for covid
            else:
                self.covid19.values[unique_id] = COVIDState.RECOVERED.value
                self.events.record_state_change((unique_id, self.model.time, self.covid19.values[unique_id]))
                self.covid_cases.record_event(
                    (unique_id, self.model.time, COVIDState.RECOVERED.value, int(False), int(False), int(False))
                )

        # If not first day, determine if individual was tested
        # Likelihood of hospitalization (and symptoms) depend on this
        else:
            tested_value = self.is_tested(unique_id)
            age = self.model.age_groups[unique_id]

            self.covid19tested.values[unique_id] = tested_value
            concurrent_conditions = self.model.concurrent_conditions[unique_id]
            p_hospitalized = self.p_hospitalized_lookup[tested_value, concurrent_conditions, age]

            p_acute = (1 - self.params["prop_of_covid_hospitalized_to_icu"]) * p_hospitalized
            p_icu = self.params["prop_of_covid_hospitalized_to_icu"] * p_hospitalized

        if self.covid19.values[unique_id] != COVIDState.RECOVERED.value:
            p_disease_severity = self.model.rng.rand()
            self.is_hospitalized(unique_id, p_disease_severity, p_icu, p_acute, location_int=hospital_int)

    def is_hospitalized(self, unique_id: int, p_disease_severity, p_icu, p_acute, location_int=None):
        """Determine if agent is hospitalized for COVID19"""
        # Mild moderate cases (don't require hospitalization, but some will still want to go)
        if p_disease_severity < (1 - p_icu - p_acute):
            self.covid19.values[unique_id] = COVIDState.COVID19MILD.value
            self.events.record_state_change((unique_id, self.model.time, self.covid19.values[unique_id]))

            # Check if agent seeks hospital care
            if location_int is None:
                location_int = self.find_location(unique_id)
            if self.model.rng.rand() < self.params["ratio_to_hospital"]:
                self.go_to_hospital(
                    unique_id=unique_id,
                    new_location=location_int,
                    covid_state=COVIDState.COVID19MILD.value,
                )
            else:
                self.covid_cases.record_event(
                    (unique_id, self.model.time, COVIDState.COVID19MILD.value, int(False), int(False), int(False))
                )

        # Critical cases
        elif p_disease_severity > 1 - p_icu:
            self.covid19.values[unique_id] = COVIDState.COVID19CRITICAL.value
            self.events.record_state_change((unique_id, self.model.time, self.covid19.values[unique_id]))

            # Assume all critical cases will try to go to an ICU
            # Some critical cases will require ventilation
            ventilator = True if self.model.rng.rand() < self.params["icu_with_ventilator_p"] else False

            if location_int is None:
                location_int = self.find_location(unique_id)
            self.go_to_hospital(
                unique_id=unique_id,
                new_location=location_int,
                covid_state=COVIDState.COVID19CRITICAL.value,
                icu=True,
                ventilator=ventilator,
            )

        # Severe cases require Non-ICU care
        else:
            self.covid19.values[unique_id] = COVIDState.COVID19SEVERE.value
            self.events.record_state_change((unique_id, self.model.time, self.covid19.values[unique_id]))

            if location_int is None:
                location_int = self.find_location(unique_id)
            # Assume all severe cases will try to go to the hospital
            self.go_to_hospital(
                unique_id=unique_id,
                new_location=location_int,
                covid_state=COVIDState.COVID19SEVERE.value,
            )

        # Set recovery day: If in hospital, use LOS - else use infection duration
        if self.model.movement.location.values[unique_id] == 0:
            self.recovery_day[unique_id] = self.model.time + self.params["infection_duration"]

    def recovery(self):
        """Have agents recover from COVID19. We assume no one has died"""

        unique_ids = [key for key, value in self.recovery_day.items() if value == self.model.time]
        self.covid19.values[unique_ids] = COVIDState.RECOVERED.value
        # No need to ranomize the order
        for unique_id in unique_ids:
            # Recovery from COVID19
            self.events.record_state_change((unique_id, self.model.time, self.covid19.values[unique_id]))
            # If currently in a hospital - keep track of NH/LT demand
            facility = self.model.movement.location.values[unique_id]
            if facility in self.model.nodes.all_hospitals:
                hospital = self.model.nodes.facilities[facility]
                bed_id = np.where(hospital.agents == unique_id)[0][0]
                # Check if ICU
                if hospital.icu_beds[bed_id]:
                    p = self.params["hospital_to_nh_icu"]
                else:
                    p = self.params["hospital_to_nh_non_icu"]

                if self.model.rng.random() < p:
                    region = self.model.unique_regions[hospital.region_str]
                    self.nh_demand_by_day[region]["daily_patients"][self.model.time] += 1

    def find_location(self, unique_id: int):
        age = self.model.age_groups[unique_id]
        county = self.model.county_codes[unique_id]
        current_location = self.model.movement.location.values[unique_id]

        # Find the first choice hospital
        p = self.model.movement.find_location_transitions(county, age, current_location)
        new_category = None
        while new_category not in ["UNC", "LARGE", "SMALL"]:
            new_category = self.model.rng_stdlib.choices(self.model.nodes.categories, cum_weights=p)[0]

        return self.model.movement.select_hospital(county, new_category, current_location)

    def regenerate_agents(self, agent_ids: np.array):
        """When an agent dies, we regenerate them. This function will prepare a new agent with CRE values"""
        # Assign susceptible CRE states
        covid19_states = np.zeros(len(agent_ids))
        covid19_states.fill(COVIDState.SUSCEPTIBLE.value)
        self.covid19.values = np.append(self.covid19.values, covid19_states).astype(np.int16)
        covid19test_states = np.zeros(len(agent_ids))
        covid19test_states.fill(COVIDTest.NA.value)
        self.covid19tested.values = np.append(self.covid19tested.values, covid19test_states).astype(np.int16)
        # Update the covid_weights: Don't worry about the age distribution, its unlikely to have changed
        covid_ages = map_array_from_dict(self.model.ages, self.covidassignment_age_dict)
        self.covid_weights = map_array_from_dict(covid_ages, self.covidassignment_weights_dict)
        # Update the testing probabilities
        self.covid19tested.probabilities = np.append(
            self.covid19tested.probabilities, self.covid19tested.find_probabilities(self.model.age_groups[agent_ids])
        )

    def find_covid_weights_by_age(self, age_params, distribution_params):
        ages = age_params
        distribution = distribution_params
        age_dict = dict()
        for i in range(0, len(ages) - 1):
            for j in range(ages[i], ages[i + 1]):
                age_dict[j] = i

        # covid_ages = np.vectorize(age_dict.get)(self.model.ages)
        # covid_ages = np.array([age_dict[age] for age in self.model.ages])
        covid_ages = map_array_from_dict(self.model.ages, age_dict)
        covidassignment_weights_dict = dict()
        for i in range(len(ages) - 1):
            covidassignment_weights_dict[i] = distribution[i]

        covid_weights = map_array_from_dict(covid_ages, covidassignment_weights_dict)

        return age_dict, covidassignment_weights_dict, covid_weights
