from collections import defaultdict
from copy import copy, deepcopy
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import expit

import src.data_input as di
from src.calibration_collection import EventStorage
from src.constants import get_multiplier, MAX_DIST
from src.facilities import Hospital
from src.misc_functions import create_cdf, normalize_and_create_cdf, random_selection
from src.state import AgeGroup, ConcurrentConditions, COVIDState, EventState, LifeState


class NorthCarolina:
    def __init__(self, model):
        self.model = model
        self.params = model.params["location"]

        # Empty Lists/Dicts to help with movement
        self.moved_agents = list()
        self.current_los = dict()
        self.leave_facility_day = dict()
        self.last_movement_day = dict()
        self.readmission_date = dict()
        self.readmission_location = dict()

        # Hospital Capacity By Day
        self.facilities = model.nodes.facilities
        self.community = model.nodes.community
        self.all_hospitals = model.nodes.all_hospitals

        # Patients Turned Away by Day
        self.patients_turned_away = EventStorage(
            column_names=["Unique_ID", "Time", "Location", "ICU", "Ventilator"], tracking=True
        )
        self.patients_completely_turned_away = EventStorage(
            column_names=["Unique_ID", "Time", "Location", "ICU", "COVID_Type"], tracking=True
        )

        # ----- Beds by Day
        self.beds_taken_by_day = dict()
        for i in self.model.unique_regions.values():
            self.beds_taken_by_day[i] = {
                "hospital_census_acute_bed": defaultdict(int),
                "hospital_census_icu_bed": defaultdict(int),
            }

        # Community Movement
        ct = di.community_transitions()
        community_dict = {(row.County_Code, row.Age_Group): row.Probability for row in ct.itertuples()}

        # Facility Movement
        ct.insert(2, "Facility", "COMMUNITY")
        ct = ct.drop(["Probability"], axis=1)
        ft = di.facility_transitions()
        self.facility_transitions = location_dict(self, ft.append(ct, sort=False))

        # Transitions
        self.transition_probabilities = dict()
        self.discharges = di.county_discharges()
        # Only hospitals in the model, please
        self.discharges = self.discharges[
            [item for item in self.discharges.columns if item in self.model.nodes.all_hospital_names]
        ]
        # Switch to ints
        self.discharges.columns = [self.model.nodes.name_to_int[item] for item in self.discharges.columns]
        # Split by category
        for item in ["UNC", "SMALL", "LARGE"]:
            discharges = self.discharges[self.model.nodes.category_ints[item]]
            self.transition_probabilities[item] = discharges.div(discharges.sum(axis=1), axis=0).fillna(0)

        # Hospitals by county: Which hospitals supply each county
        dis = self.discharges
        self.hospitals_by_county = {county: list(dis.columns[dis.loc[county] > 0]) for county in list(dis.index)}

        self.add_distributions_for_transfers()

        self.county_to_hospital_distances = di.county_hospital_distances()

        self.county_nh_probabilities = di.county_facility_distribution(
            "nh",
            distance_weight=self.params["facilities"]["NH"]["location_probabilities"]["distance_weight"],
            bed_count_weight=self.params["facilities"]["NH"]["location_probabilities"]["bed_count_weight"],
            closest_n=self.params["facilities"]["NH"]["location_probabilities"]["closest_n"],
        )
        self.county_ltach_probabilities = di.county_facility_distribution(
            "ltach",
            distance_weight=self.params["facilities"]["LT"]["location_probabilities"]["distance_weight"],
            bed_count_weight=self.params["facilities"]["LT"]["location_probabilities"]["bed_count_weight"],
            closest_n=self.params["facilities"]["LT"]["location_probabilities"]["closest_n"],
        )
        self.county_hospital_probabilities = di.county_facility_distribution(
            "hospital",
            distance_weight=self.params["facilities"]["LARGE"]["location_probabilities"]["distance_weight"],
            bed_count_weight=self.params["facilities"]["LARGE"]["location_probabilities"]["bed_count_weight"],
            closest_n=self.params["facilities"]["LARGE"]["location_probabilities"]["closest_n"],
        )

        self.nh_los = self.make_nh_los()

        self.location = EventState(enum=None, transition_dict=community_dict, key_types=["County", AgeGroup])
        column_names = ["Unique_ID", "Time", "Location", "New Location", "ICU"]
        self.location.events = EventStorage(column_names, tracking=self.model.track_events)
        self.location.probabilities = self.location.find_probabilities(
            keys=list(zip(self.model.county_codes, self.model.age_groups))
        )
        self.location.initiate_values(count=len(self.model.population), value=self.community)
        self._agents_in_hospitals = np.full(shape=len(self.model.population), fill_value=False, dtype=bool)
        # Update the community movement probability based on the concurrent conditions
        self.location.update_community_probability(
            cp=self.location.probabilities, age=self.model.age_groups, cc=self.model.concurrent_conditions
        )

        # ----- Assign LOS based on starting location and update location values
        self.sampled_lt_los = self.init_los_from_gamma(self.params["facilities"]["LT"]["los"])
        self.init_nh_ltach_agents()
        self.init_hospitals()
        self.location.copy_values_to_previous()

        # Census of patients turned away who should be in
        # a hospital bed because they are still sick
        self.pta_need_bed = dict()
        for i in self.model.unique_regions.values():
            self.pta_need_bed[i] = {"pta_census_acute_bed": defaultdict(int), "pta_census_icu_bed": defaultdict(int)}

        # Account for readmissions:

    def step(self):
        """Step through 1 day of location updates"""
        self.moved_agents = []
        # check for covid recovery before moving agents
        self.model.disease.recovery() if self.model.run_covid else None
        self.model.disease.step() if self.model.run_covid else None
        self.facility_movement()
        self.community_movement()
        # self.readmission_movement()  # NOTE: Readmission is disabled while we figure out ramp-up issues.

    def init_nh_ltach_agents(self):
        """For agents starting in NH and LTs, determing LOS and assign agent to facility"""
        # sample people from counties that are closest to the facilities to fill the capacities
        ltach_to_county_p = di.facility_to_county_probabilities("ltach")
        # convert to int
        for key in list(ltach_to_county_p.keys()):
            ltach_to_county_p[self.model.nodes.name_to_int[key]] = ltach_to_county_p.pop(key)
        nh_to_county_p = di.facility_to_county_probabilities("nh")
        # convert to int
        for key in list(nh_to_county_p.keys()):
            nh_to_county_p[self.model.nodes.name_to_int[key]] = nh_to_county_p.pop(key)

        multiplier = get_multiplier(self.model.params)

        nh_county_counts = {c: [] for c in range(1, 201, 2)}
        lt_county_counts = {c: [] for c in range(1, 201, 2)}
        for location in self.model.nodes.category_ints["NH"]:
            facility = self.facilities[location]
            prob_list = nh_to_county_p[location]
            beds = max(1, int(round((facility.beds * multiplier * 0.7))))
            counties = self.model.rng_stdlib.choices(prob_list[1], cum_weights=prob_list[0], k=beds)
            for county in counties:
                nh_county_counts[county].append(location)
        for location in self.model.nodes.category_ints["LT"]:
            facility = self.facilities[location]
            prob_list = ltach_to_county_p[location]
            beds = max(1, int(round((facility.beds * multiplier * 0.7))))
            counties = self.model.rng_stdlib.choices(prob_list[1], cum_weights=prob_list[0], k=beds)
            for county in counties:
                lt_county_counts[county].append(location)

            # For each person, sample an actual person in the community from the county
        for county in nh_county_counts:
            locations = nh_county_counts[county]
            if len(locations) == 0:
                continue
            unique_ids = self.model.unique_ids[(self.model.county_codes == county) & (self.location.values == 0)]
            unique_ids = unique_ids[self.model.age_groups[unique_ids] == 2]

            if len(unique_ids) > 0 and len(locations) <= len(unique_ids):
                unique_ids = self.model.rng.choice(unique_ids, size=len(locations), replace=False)
            for _, unique_id in enumerate(unique_ids):
                location = locations[_]
                self.assign_los(unique_id=unique_id, new_location=location, initialize=True)
                self.location.values[unique_id] = location
                self.facilities[location].add_agent(unique_id)

        for county in lt_county_counts:
            locations = lt_county_counts[county]
            if len(locations) == 0:
                continue
            unique_ids = self.model.unique_ids[(self.model.county_codes == county) & (self.location.values == 0)]
            if len(unique_ids) > 0 and len(locations) <= len(unique_ids):
                unique_ids = self.model.rng.choice(unique_ids, size=len(locations), replace=False)
            for _, unique_id in enumerate(unique_ids):
                location = locations[_]
                self.assign_los(
                    unique_id=unique_id, new_location=location, los=self.model.rng.choice(self.sampled_lt_los)
                )
                self.location.values[unique_id] = location
                self.facilities[location].add_agent(unique_id)

    def init_los_from_gamma(self, los: dict):
        """Initialize some length of stays to sample from for a unc hospital
        according to the following steps:

        1. Initialize a large number of los from the gamma distribution
        2. Calculate how many days it would take to stabalize the los by getting
        the 95th percentile of the distribution
        3. Step the LOS forward that many days. When a LOS reaches 0, sample from the
        distribution for a new number to replace it

        """

        initial_sample = self.model.rng.gamma(los["shape"], los["support"], size=10000)
        initial_sample = np.rint(initial_sample)

        num_days = int(np.percentile(initial_sample, 95))

        sample = pd.Series(initial_sample)
        for day in range(num_days):
            sample -= 1
            a_filter = sample < 0
            sample[a_filter] = self.model.rng_stdlib.choices(initial_sample, k=sample[a_filter].size)
        return sample.tolist()

    def init_hospitals(self):
        """For agents not starting in the community, determine LOS and ICU assignment."""
        # A dictionary to hold the facilities needing someone from this county/age combination
        ca_normal = dict()
        for county in range(1, 201, 2):
            for age in [0, 1, 2]:
                ca_normal[(county, age)] = []
        ca_icu = deepcopy(ca_normal)

        # for each category in the params with a distribution, sample once
        sampled_los = {}
        for key in self.params["facilities"]:
            if "los" in self.params["facilities"][key]:
                los = self.params["facilities"][key]["los"]
                sampled_los[key] = self.init_los_from_gamma(los)

        for h_int in self.all_hospitals:
            hospital = self.facilities[h_int]

            if hospital.name in sampled_los.keys():
                hospital.los_for_sampling = sampled_los[hospital.name]
            elif hospital.category in sampled_los.keys():
                hospital.los_for_sampling = sampled_los[hospital.category]
            else:
                hospital.los_for_sampling = []

            select_facility(self, hospital=hospital, ca_dict=ca_normal, icu=False)
            select_facility(self, hospital=hospital, ca_dict=ca_icu, icu=True)

        self._init_vent_prob = self.model.rng.rand(len(self.location.values)) < self.params["ventilator_probability"]
        self._init_community_agents = self.location.values == 0
        # Find appropriate agents
        for key, value in ca_normal.items():
            self.assign_agents(key, value, icu=False)
        for key, value in ca_icu.items():
            self.assign_agents(key, value, icu=True)

    def assign_agents(self, key: int, values: list, icu: bool):
        county_code = key[0]
        age_group = key[1]

        filter1 = self.model._county_code_mask[county_code]
        filter2 = self.model.age_groups == age_group
        filter3 = self._init_community_agents
        unique_ids = self.model.unique_ids[filter1 & filter2 & filter3].tolist()
        # Sample IDS
        if len(unique_ids) > 0:
            unique_ids = self.model.rng_stdlib.sample(list(unique_ids), k=len(values))
            for i, unique_id in enumerate(unique_ids):
                self.model.icu_status[unique_id] = int(icu)
                location_int = values[i]
                facility = self.facilities[location_int]
                if len(facility.los_for_sampling) > 0:
                    sampled_los = self.model.rng_stdlib.choices(facility.los_for_sampling)[0]
                    self.assign_los(unique_id=unique_id, new_location=location_int, los=sampled_los)
                else:
                    self.assign_los(unique_id=unique_id, new_location=location_int, initialize=True)
                self.location.values[unique_id] = location_int
                self._agents_in_hospitals[unique_id] = location_int in self.all_hospitals
                if icu:
                    need_ventilator = self._init_vent_prob[unique_id]
                    if facility.add_agent(unique_id, icu=icu, ventilator=need_ventilator):
                        continue
                    else:
                        facility.add_agent(unique_id, icu=icu, ventilator=~need_ventilator)
                else:
                    facility.add_agent(unique_id, icu=icu)

    def community_movement(self):
        """Find all agents in the community and see which ones are selected to move. Then move those agents"""
        living = self.model.life.values == LifeState.ALIVE
        community = self.location.values == self.community
        use_agents = self.model.unique_ids[living & community]
        probabilities = self.location.probabilities[use_agents]
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        unique_ids = use_agents[selected_agents]
        self.model.rng.shuffle(unique_ids)
        for unique_id in unique_ids:
            self.location_update(unique_id)

    def facility_movement(self):
        """Move all agents not in the community whose LOS ends today"""
        unique_ids = [key for key, value in self.leave_facility_day.items() if value == self.model.time]
        self.model.rng.shuffle(unique_ids)
        for unique_id in unique_ids:
            self.location_update(unique_id)

    def location_update(self, unique_id: int):
        """Update an agents location. If new_category is specified, the agent will move to the given category.
        Otherwise, the model will select the new category based on probabilities.
        """
        current_location = self.location.values[unique_id]
        current_category = self.facilities[current_location].category
        previous_category = self.facilities[self.location.previous[unique_id]].category

        age = self.model.age_groups[unique_id]
        county = self.model.county_codes[unique_id]
        self.uid = unique_id

        # 80% of previous NH patients (who are leaving a hospital) must return to a NH
        if (previous_category == "NH") and (current_category in self.model.nodes.hospital_categories):
            if self.model.rng.rand() < self.params["tuning"]["NH_to_ST_to_NH"]:
                new_category = "NH"
            else:
                new_category = self.non_nh_movement(county, age, current_location)
        else:
            p = self.find_location_transitions(county, age, current_location)
            new_category = self.model.rng_stdlib.choices(self.model.nodes.categories, cum_weights=p)[0]

        # Go home
        if new_category == "COMMUNITY":
            # If currently in an STACH, prepare agent for possible readmission
            if current_category in self.model.nodes.hospital_categories:
                if self.model.rng.rand() < self.params["readmission"]["rate"]:
                    days = self.params["readmission"]["days"]
                    self.readmission_date[unique_id] = self.model.time + self.model.rng.randint(2, days)
                    self.readmission_location[unique_id] = current_location
            self.go_home(unique_id, current_location)
        # Move to a hospital
        if new_category in self.model.nodes.hospital_categories:
            new_location = self.select_hospital(county=county, category=new_category, location=current_location)
            self.move_to_stach(
                unique_id=unique_id, current_location=current_location, new_location=new_location, test_icu=True
            )
        # Move to a NH
        elif new_category == "NH":
            new_location = self.select_nh(county)
            self.move_agent(
                unique_id=unique_id, current_location=current_location, new_location=new_location, assign_los=True
            )
        # Move to an LT
        elif new_category == "LT":
            new_location = self.select_lt(county)
            self.move_agent(
                unique_id=unique_id, current_location=current_location, new_location=new_location, assign_los=True
            )

    def move_to_stach(
        self,
        unique_id: int,
        current_location: int,
        new_location: int,
        covid_state: int = COVIDState.SUSCEPTIBLE.value,
        force_icu: bool = False,
        test_icu: bool = True,
        ventilator: bool = False,
    ):
        """Move patient to a STACH. This function determines if an ICU or non-ICU bed will be used"""
        # Demographics
        age = self.model.age_groups[unique_id]
        size = self.facilities[new_location].n_total_beds
        cc = self.model.concurrent_conditions[unique_id]
        potential_los = self.potential_los(unique_id, new_location)

        # -----  New location must have ICU beds
        icu = False
        if test_icu:
            if self.facilities[new_location].n_icu_beds > 0:
                p = self.find_icu_probability(age, size, potential_los, cc) * self.params["icu_reduction_multiplier"]
                if self.model.rng.rand() < p:
                    icu = True

        if force_icu | icu:
            icu = True
            if self.model.rng.rand() < self.params["ventilator_probability"]:
                ventilator = True

        self.stach_movement(
            unique_id=unique_id,
            current_location=current_location,
            new_location=new_location,
            los=potential_los,
            icu=icu,
            ventilator=ventilator,
            covid_state=covid_state,
        )

    def stach_movement(
        self,
        unique_id: int,
        current_location: int,
        new_location: int,
        los: int,
        covid_state: int = COVIDState.SUSCEPTIBLE.value,
        icu: bool = False,
        ventilator: bool = False,
    ):
        """Move the patient to a normal hospital bed.
        - If first choice doesn't work - record this
        - Try one of 3 additional options
        """
        # If space, move them to the hospital
        if self.facilities[new_location].add_agent(unique_id, icu=icu, ventilator=ventilator):
            self.move_agent(
                unique_id=unique_id,
                current_location=current_location,
                new_location=new_location,
                assign_los=False,
                icu=icu,
            )
            self.assign_los(unique_id=unique_id, new_location=new_location, los=los)
        # If not and patient is a transfer, send them home (there was no room for them)
        elif current_location != self.community:
            self.go_home(unique_id, current_location=current_location)
        # Person is turned away - Try to find an equal bed somewhere else
        else:
            self.patients_turned_away.record_event(
                (unique_id, self.model.time, new_location, int(icu), int(ventilator))
            )

            # Now Try one of three options:
            # #1: An Agent's Second Choice Hospital
            move = False
            county = self.model.county_codes[unique_id]
            new_location2 = self.select_hospital(county, self.facilities[new_location].category, new_location)
            if self.facilities[new_location2].add_agent(unique_id, icu=icu, ventilator=ventilator):
                move = True
                final_loc = new_location2

            # #2: Any hospital the agent would consider
            else:
                ints = copy(self.hospitals_by_county[county])
                for item in [new_location, new_location2]:
                    if item in ints:
                        ints.remove(item)
                for item in ints:
                    if self.facilities[item].add_agent(unique_id, icu=icu, ventilator=ventilator):
                        final_loc = item
                        move = True
                        break

                if not move:
                    # #3: Any other hospital by distance from county centroid
                    facilities_by_distance = [
                        f for f in self.county_to_hospital_distances[county] if f["distance_mi"] <= MAX_DIST
                    ]
                    for facility in facilities_by_distance:
                        try:
                            facility_int = self.model.nodes.name_to_int[facility["Name"]]
                            if facility_int not in ints:
                                if self.facilities[facility_int].add_agent(unique_id, icu=icu, ventilator=ventilator):
                                    final_loc = facility_int
                                    move = True
                                    break
                        except Exception as E:
                            E
                            continue

            if move:
                self.move_agent(
                    unique_id=unique_id,
                    current_location=current_location,
                    new_location=final_loc,
                    assign_los=False,
                    icu=icu,
                )
                self.assign_los(unique_id=unique_id, new_location=new_location, los=los)
            else:
                self.patients_completely_turned_away.record_event(
                    (unique_id, self.model.time, new_location, int(icu), covid_state)
                )

    def readmission_movement(self):
        """Model readmission from the community. Only move patient if they are still at home.
        Some patients will return before their readmission date - they should not be moved.
        """
        unique_ids = [key for key, value in self.readmission_date.items() if value == self.model.time]
        self.model.rng.shuffle(unique_ids)
        for unique_id in unique_ids:
            if self.location.values[unique_id] == 0:
                if self.model.life.values[unique_id] == 0:
                    new_location = self.readmission_location[unique_id]
                    selected_los = self.potential_los(unique_id, new_location)
                    self.stach_movement(
                        unique_id=unique_id, current_location=0, new_location=new_location, los=selected_los
                    )
            del self.readmission_date[unique_id]
            del self.readmission_location[unique_id]

    @lru_cache(maxsize=None)
    def find_icu_probability(self, age: int, size: int, current_los: int, cc: int):
        """Calculate the probability of a specific agent going to an ICU"""
        logit = -2.4035
        if age == AgeGroup.AGE1:
            logit += 0.1395
        elif age == AgeGroup.AGE2:
            logit += 0.1326
        if size > 400:
            logit += 0.1867
        if cc == ConcurrentConditions.YES:
            logit += 0.8169
        if current_los <= 7:
            pass
        elif current_los <= 30:
            logit += 0.2571
        else:
            logit += 0.7337
        return expit(logit)

    def move_agent(self, unique_id: int, current_location: int, new_location: int, assign_los: bool, icu: bool = False):
        """Move an agent from their current_location to a new location"""
        self.last_movement_day[unique_id] = self.model.time
        self.location.events.record_state_change((unique_id, self.model.time, current_location, new_location, int(icu)))
        # Normal may have been assigned to ICU:
        self.model.icu_status[unique_id] = int(icu)

        if assign_los:
            self.assign_los(unique_id=unique_id, new_location=new_location)

        # Finalize Movement
        self.facilities[current_location].remove_agent(unique_id)
        self.moved_agents.append(unique_id)
        self.location.previous[unique_id] = current_location
        self.location.values[unique_id] = new_location
        self._agents_in_hospitals[unique_id] = new_location in self.all_hospitals
        self.update_death_probabilities(unique_id)

    def go_home(self, unique_id: int, current_location: int):
        """Send a patient to the community"""
        self.facilities[current_location].remove_agent(unique_id)

        # Update their LOS based on how long they were actually at the facility (imporant for people who die)
        self.current_los[unique_id] = self.model.time - self.last_movement_day.get(unique_id, 0)
        self.move_agent(
            unique_id=unique_id, current_location=current_location, new_location=self.community, assign_los=False
        )
        del self.leave_facility_day[unique_id]
        del self.current_los[unique_id]

    def non_nh_movement(self, county, age, current_location, count=0):
        """Force movement to something other than a NH"""
        new_category = "NH"
        p = self.find_location_transitions(county, age, current_location)
        while new_category == "NH":
            new_category = self.model.rng_stdlib.choices(self.model.nodes.categories, cum_weights=p)[0]
            count += 1
            if count > 20:
                raise ValueError("While loop continues indefinetly.")
        return new_category

    def potential_los(self, unique_id: int, new_location: int):
        """LOS needs to be known before ICU status can be assigned. However, the ICUs could be full.
        Calculate a potential LOS (perhaps based on symptoms) to determine if an ICU will be needed.
        """
        location_category = self.facilities[new_location].category
        # UNC has distributions by location, all other facilities use their category
        if location_category == "UNC":
            los = self.params["facilities"][self.facilities[new_location].name]["los"]
        else:
            los = self.params["facilities"][location_category]["los"]
        # Pick a random LOS based on the distribution matching the location
        if los["distribution"] == "Gamma":
            selected_los = int(round(self.model.rng.gamma(los["shape"], los["support"]), 0))
        else:
            raise ValueError("LOS distribution of type {} is not supported.".format(los["distribution"]))
        # LOS cannot be 0 days. They must stay at location at least one day
        if selected_los == 0:
            selected_los += 1

        return selected_los

    def assign_los(self, unique_id: int, new_location: int, initialize: bool = False, los: int = None):
        """ Given a new_location, select a LOS for a new patient """
        # If at home, do nothing
        if new_location == self.community:
            return
        # If LOS was pretermined, use it
        if los is not None:
            self.current_los[unique_id] = los
            self.leave_facility_day[unique_id] = self.model.time + self.current_los[unique_id]
            return

        location_category = self.facilities[new_location].category

        # If NH: randomly select LOS from possible LOS
        if location_category == "NH":
            self.current_los[unique_id] = self.nh_los["LOS"][self.model.rng.randint(0, len(self.nh_los["LOS"]))]
        else:
            selected_los = self.potential_los(unique_id, new_location)
            self.current_los[unique_id] = selected_los

        if initialize:
            # Instead of using LOS, we use the "Time Until Leaving" distribution
            if location_category == "NH":
                a_list = self.nh_los["Time_Until_Leaving"]
                self.current_los[unique_id] = a_list[self.model.rng.randint(0, len(a_list))]
        # Set the leave day
        self.leave_facility_day[unique_id] = self.model.time + self.current_los[unique_id]

    def select_nh(self, county: int):
        """ Select a new NH """
        return self.model.nodes.name_to_int[
            random_selection(
                self.model.rng.rand(), self.county_nh_probabilities[county][0], self.county_nh_probabilities[county][1]
            )
        ]

    def select_lt(self, county: int):
        """ Select a new LTACH """
        return self.model.nodes.name_to_int[
            random_selection(
                self.model.rng.rand(),
                self.county_ltach_probabilities[county][0],
                self.county_ltach_probabilities[county][1],
            )
        ]

    def select_hospital(self, county: int, category: str, location: int) -> int:
        """ Select a random stach """
        options = self.model.nodes.category_ints[category]
        p = self.facilities[location].options[category][county]
        if sum(p) > 0:
            return self.model.rng_stdlib.choices(options, cum_weights=p)[0]
        else:
            # select a LARGE hospital based on distance and bed counts
            if category == "LARGE":
                new_location = location
                options = self.county_hospital_probabilities[county][1]
                p = self.county_hospital_probabilities[county][0]
                count = 0
                while new_location == location:
                    if count > 100:
                        raise ValueError("While loop continues indefinetly.")
                    name = self.model.rng_stdlib.choices(options, cum_weights=p)[0]
                    new_location = self.model.nodes.name_to_int[name]
                    count += 1
                return new_location
            # For UNC we only send to one of the two large hospitals
            if category == "UNC":
                options = self.model.nodes.large_unc
            # Pick a random UNC or a random SMALL based on the category
            return self.model.rng.choice([h_int for h_int in options if h_int != location])

    def make_nh_los(self):
        nh_los = di.nh_los()
        nh_los = nh_los[(0 < nh_los.los) & (nh_los.los < 2000)].copy()
        nh_los = nh_los.astype(int)
        nh_los.cfreq = nh_los.cfreq.apply(lambda x: int(x * 0.1))
        a_list = []
        for row in nh_los.itertuples():
            a_list.extend([row.los] * int(row.cfreq))

        nh_los2 = di.nh_los2()

        nh_dict = dict()
        nh_dict["LOS"] = a_list
        nh_dict["Time_Until_Leaving"] = [item[0] for item in nh_los2.values]
        return nh_dict

    @lru_cache(maxsize=None)
    def find_location_transitions(self, county: int, age: int, loc_int: int) -> float:
        return list(self.facility_transitions[(county, age, loc_int)])

    def add_distributions_for_transfers(self):
        for h_int in self.facilities:
            hospital = self.facilities[h_int]
            hospital.options = {"LARGE": {}, "SMALL": {}, "UNC": {}}

        for county in range(1, 201, 2):
            arrays = {}
            arrays["LARGE"] = self.transition_probabilities["LARGE"].loc[county].values
            arrays["SMALL"] = self.transition_probabilities["SMALL"].loc[county].values
            arrays["UNC"] = self.transition_probabilities["UNC"].loc[county].values
            normalized_arrays = {}
            normalized_arrays["LARGE"] = normalize_and_create_cdf(arrays["LARGE"])
            normalized_arrays["SMALL"] = normalize_and_create_cdf(arrays["SMALL"])
            normalized_arrays["UNC"] = normalize_and_create_cdf(arrays["UNC"])

            for h_int in self.facilities:
                hospital = self.facilities[h_int]
                for category in ["LARGE", "SMALL", "UNC"]:
                    if h_int in self.model.nodes.category_ints[category]:
                        temp_array = copy(arrays[category])
                        temp_array[self.model.nodes.category_ints[category].index(h_int)] = 0
                        hospital.options[category][county] = normalize_and_create_cdf(temp_array)
                    else:
                        hospital.options[category][county] = normalized_arrays[category]

    def regenerate_agents(self, agent_ids: list):
        """For agents who have died, create new agents to take their place"""
        new_locations = [self.model.nodes.community] * len(agent_ids)
        new_ids = ["COMMUNITY"] * len(agent_ids)
        # Find everyone's probability of dying
        self.model.life.probabilities = np.append(
            self.model.life.probabilities,
            self.model.life.find_probabilities(list(zip(self.model.age_groups[agent_ids], new_ids))),
        )
        # Everyone being created should be alive
        self.model.life.values = np.append(self.model.life.values, [LifeState.ALIVE.value] * len(agent_ids)).astype(
            np.int16
        )

        # Find everyone's probability of leaving the community
        keys = list(zip(self.model.county_codes[agent_ids], self.model.age_groups[agent_ids]))
        self.location.probabilities = np.append(self.location.probabilities, self.location.find_probabilities(keys))
        # Everyone should start in the community
        self.location.values = np.append(self.location.values, new_locations).astype(np.int16)
        self.location.copy_values_to_previous()
        self._agents_in_hospitals = np.append(self._agents_in_hospitals, np.full(len(new_locations), False))

    def update_death_probabilities(self, unique_id: int):
        """As an agent changes locations, they need their death probability update"""
        age = self.model.age_groups[unique_id]
        location_int = self.location.values[unique_id]
        location_category = self.facilities[location_int].category
        self.model.life.probabilities[unique_id] = self.model.life.transition_dict[age, location_category]


def location_dict(location, lt):
    ld = {}
    county_code_loc = lt.columns.get_loc("County_Code")
    age_loc = lt.columns.get_loc("Age_Group")
    facility_loc = lt.columns.get_loc("Facility")
    community_loc = lt.columns.get_loc("COMMUNITY")
    for item in lt.values:
        key_0 = item[county_code_loc]
        key_1 = item[age_loc]
        if item[facility_loc] == "COMMUNITY":
            cdf = normalize_and_create_cdf(item[community_loc:])
            ld[(key_0, key_1, 0)] = cdf  # TODO, fix the 0
        elif item[facility_loc] == "NH":
            cdf = create_cdf(item[community_loc:])
            # There is only one row for NH - apply to all NHs though
            for key_2 in location.model.nodes.category_ints["NH"]:
                ld[(key_0, key_1, key_2)] = cdf
        elif item[facility_loc] == "LT":
            # There is only one row for LT - apply to all LTs though
            cdf = create_cdf(item[community_loc:])
            for key_2 in location.model.nodes.category_ints["LT"]:
                ld[(key_0, key_1, key_2)] = cdf
        else:
            key_2 = location.model.nodes.name_to_int[item[facility_loc]]
            ld[(key_0, key_1, key_2)] = create_cdf(item[community_loc:])
    return ld


def select_facility(movement, hospital: Hospital, ca_dict: dict, icu: bool):
    """Append the number of hospital patients required for a given county/age dictionary.
    This will help fill the beds with agents
    """
    rows = pd.DataFrame(movement.discharges[hospital.model_int])
    rows.loc[:, "Percentage"] = rows[hospital.model_int] / rows[hospital.model_int].sum()
    beds = int(round(hospital.n_normal_beds / max(hospital.base_n_normal_beds, 1) * hospital.initial_normal))
    if icu:
        beds = int(round(hospital.n_icu_beds / max(hospital.base_n_icu_beds, 1) * hospital.initial_icu))

    p = rows.Percentage.values
    for bed in range(0, beds):
        # --- Randomly select a county based on the percentage
        county = movement.model.rng_stdlib.choices(rows.index, weights=p)[0]
        # --- Randomly select an age based on 40% <50, 20% 50-65, and 40% 65+
        age = movement.model.rng_stdlib.choices([0, 1, 2], cum_weights=[0.4, 0.6, 1.0])[0]
        ca_dict[(county, age)].append(hospital.model_int)
