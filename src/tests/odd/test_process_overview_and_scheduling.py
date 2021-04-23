# * Identify which agents die. If an agent dies:
#     * Record a life state change
#     * For agents in a non-community location, send the agent to the community to open up the bed that they were occupying.
#     * Add the agent to the list of agents to recreate.

from copy import copy

import numpy as np
import pytest
from src.calibration_collection import init_beds_taken_data, step_beds_taken_data
from src.constants import get_multiplier
from src.data_processing import county_to_coalition_map
from src.state import COVIDState, COVIDTest, LifeState
from src.tests.fixtures import model, model_function_scoped

county_map = county_to_coalition_map()
ldm_copy = model_function_scoped


class TestLifeStep:
    def test_life_step_non_community(self, ldm_copy):
        """
        * For agents in a non-community location, send the agent to the community to open up the bed that they were occupying.
        * Add the agent to the list of agents to recreate.
        """
        # arrange
        ldm = ldm_copy
        agent_locations = ldm.movement.location.values
        community_agent_id = next(id_ for id_, location in enumerate(agent_locations) if location != 0)

        # act
        ldm.life.life_update(community_agent_id)

        # assert
        assert ldm.life.values[community_agent_id] == LifeState.DEAD.value, "Agent state not DEAD"
        new_agent_locations = ldm.movement.location.values
        assert new_agent_locations[community_agent_id] == 0, "Agent not added to community"
        assert community_agent_id in ldm.life.agents_to_recreate, "Agent not added to recreation list"

    def test_life_step_community(self, ldm_copy):
        """
        Test life step for agent currently in community
        """

        ldm = ldm_copy
        agent_locations = ldm.movement.location.values
        community_agent_id = next(id_ for id_, location in enumerate(agent_locations) if location == 0)
        ldm.life.life_update(community_agent_id)
        assert ldm.life.values[community_agent_id] == LifeState.DEAD.value, "Agent state not DEAD"
        new_agent_locations = ldm.movement.location.values
        assert new_agent_locations[community_agent_id] == 0, "Agent not added to community"
        assert community_agent_id in ldm.life.agents_to_recreate, "Agent not added to recreation list"


class TestLocationStep:
    def test_covid_recovery(self, ldm_copy):
        "* Assess which COVID-19 patients are set to recover. If they recover, preform a COVID-19 state change"
        ldm = ldm_copy
        days = 14  # recovery is 14 days
        for day in range(days):
            ldm.time = day
            ldm.step()
        agents_recovering = [key for key, value in ldm.disease.recovery_day.items() if value == ldm.time]
        assert len(agents_recovering) > 1
        assert all((state != COVIDState.RECOVERED.value for state in agents_recovering))
        ldm.step()
        ldm.time += 1
        new_covid_states = ldm.disease.covid19.values[agents_recovering]
        assert all((state == COVIDState.RECOVERED.value for state in new_covid_states))

    def test_location_update(self, ldm_copy):
        "* Administer a location update (outlined in Section 7. Submodels) for any agent whose LOS ends on the current day"
        ldm = ldm_copy
        ldm.step()
        ldm.time += 1
        agents_leaving = [
            id_ for id_, leave_facility_day in ldm.movement.leave_facility_day.items() if leave_facility_day == ldm.time
        ]
        current_locations = copy(ldm.movement.location.values[agents_leaving])
        ldm.step()
        ldm.time += 1
        new_locations = ldm.movement.location.values[agents_leaving]
        assert (new_locations == current_locations).sum() == 0, (
            f"{(new_locations == current_locations).sum()} of {len(current_locations)}"
            " expected agents did not relocate."
        )

    def test_leave_community_update(self, ldm_copy):
        "* Administer a location update for any agent selected to leave the community"
        ldm = ldm_copy
        ldm.step()
        ldm.time += 1

        living = ldm.life.values == LifeState.ALIVE
        community = ldm.movement.location.values == 0
        use_agents = ldm.unique_ids[living & community][:100]  # use only 100 otherwise this is SLOW
        for unique_id in use_agents:
            ldm.movement.location_update(unique_id)

        assert (ldm.movement.location.values[use_agents] == 0).sum() == 0, "Not all agents moved out of community"

    @pytest.mark.skip("TODO")
    def test_readmission_update(self, ldm_copy):
        "* Administer a location update for any agent whose readmission date is for the current day"
        ldm = ldm_copy
        for day in range(10):
            ldm.step()
            ldm.time += 1
        agents_readmitted = [id_ for id_, date in ldm.movement.readmission_date.items() if date == ldm.time]
        current_locations = copy(ldm.movement.location.values[agents_readmitted])
        ldm.step()
        ldm.time += 1
        new_locations = ldm.movement.location.values[agents_readmitted]
        assert (new_locations == current_locations).sum() == 0, (
            f"{(new_locations == current_locations).sum()} of {len(current_locations)}"
            " expected agents did not experience readmission."
        )


class TestCOVID19Step:
    def test_infection_count(self, ldm_copy):
        "* Use the SEIR case projections by county to select susceptible agents by county to be newly infected"
        ldm = ldm_copy
        ldm.time = 0
        ldm.step()

        scaler = get_multiplier(ldm.params)
        initial_live = ldm.disease.initial_live_infections * scaler
        initial_cumul = ldm.disease.initial_cumulative_infections * scaler

        covid_events = ldm.disease.covid_cases.make_events()
        # assert the initial total number of cases is within 10% of estimates
        assert (
            abs(len(covid_events) - initial_cumul) / initial_cumul
        ) <= 0.1, f"The cumulative cases initialized is outside of 10% of the actual value, TRUE:{len(covid_events)}, ACTUAL:{initial_cumul}"
        # the initial live cases is within 10% of estimates
        assert (
            abs(len(covid_events.loc[covid_events["Type"] < 4]) - initial_live) / initial_live
        ) <= 0.1, f"The live cases initialized is outside of 10% of the actual value, TRUE:{len(covid_events.loc[covid_events['Type'] < 4])}, ACTUAL:{initial_live}"

    def test_infected_testing(self, ldm_copy):
        "* Estimate probability of being tested among newly infected agents"
        # for the newly infected agents the percent tested should be close to the parameter
        ldm = ldm_copy
        ldm.time = 0
        ldm.step()
        # need to step one more day for this test. Testing is ignored on first day
        ldm.time += 1
        ldm.step()
        covid_events = ldm.disease.covid_cases.make_events()

        day2_ids = covid_events.loc[covid_events["Time"] == ldm.time]["Unique_ID"].astype(int).tolist()
        tested = ldm.disease.covid19tested.values
        for unique_id in day2_ids:
            assert tested[unique_id] in [
                COVIDTest.TESTED.value,
                COVIDTest.UNTESTED.value,
            ], "New case for second day of covid is neither tested or untested"

    def test_symptom_severity(self, ldm_copy):
        "* Assign symptom severity for newly infected agents"
        # the newly infected agents should have varying severity
        ldm = ldm_copy
        ldm.time = 0
        ldm.step()
        covid_events = ldm.disease.covid_cases.make_events()
        assert (
            len(covid_events["Type"].unique()) == 4
        ), f"The number of unique symptom severity scores is {len(covid_events['Type'].unique())} != 4"

    @pytest.mark.skip("TODO")
    def test_infected_hospitalization(self, ldm_copy):
        "* Determine hospitalization among infected agents"
        "* Hospitalize agents and assign length of stay"
        ldm = ldm_copy
        ldm.time = 0
        ldm.step()

        # move the hosp counts for day 0 here
        scaler = get_multiplier(ldm.params)
        unscaled_hosp_dict = ldm.disease.coalition_hosp
        scaled_hosp_dict = {key: round(val * scaler) for key, val in unscaled_hosp_dict.items()}

        # get the actual number of hospitalizations per coalition
        covid_events = ldm.disease.covid_cases.make_events()
        coalition = np.array(list(map(lambda x: county_map[x], ldm.county_codes)))

        for co in scaled_hosp_dict:
            people_in_co = ldm.unique_ids[(coalition == co)]
            co_events = covid_events.loc[
                (covid_events["Seeking Hospital"] == 1) & (covid_events["Unique_ID"].isin(people_in_co))
            ]
            if len(co_events) > 30:
                assert (
                    abs(len(co_events) - scaled_hosp_dict[co]) / scaled_hosp_dict[co] <= 0.5
                ), f"The number of hospitalizations is out of bounds, {len(co_events)} instead of {scaled_hosp_dict[co]}"
            else:
                assert (
                    abs(len(co_events) - scaled_hosp_dict[co]) <= 10
                ), f"The number of hospitalizations is out of bounds, {len(co_events)} instead of {scaled_hosp_dict[co]}"

        # critical or severe patients should either be in a hospital or be in the patient turned away event list
        hosp_seekers = covid_events.loc[covid_events["Seeking Hospital"] == 1]["Unique_ID"].astype(int).tolist()
        turned_away = ldm.movement.patients_turned_away.make_events()["Unique_ID"].astype(int).tolist()
        for unique_id in hosp_seekers:
            assert (ldm.movement.location.values[unique_id] in ldm.movement.all_hospitals) or (
                unique_id in turned_away
            ), f"Person {unique_id} was neither hospitalized or turned away"
