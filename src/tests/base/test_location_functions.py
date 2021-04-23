from copy import deepcopy

import pytest
import src.data_input as di
from src.constants import get_multiplier
from src.facilities import Community, Hospital
from src.state import LifeState
from src.tests.fixtures import model


def test_locations_class(model):
    m = deepcopy(model)
    hospitals = di.hospitals()

    # Test Community
    assert m.nodes.community == 0
    assert isinstance(m.nodes.facilities[0], Community)

    # UNC has same number of hospitals in hospitals file
    unc_hospitals = [h for i, h in m.nodes.facilities.items() if h.category == "UNC"]
    assert len(unc_hospitals) == len(hospitals[hospitals.Category == "UNC"])

    # Categories must be in alignment
    for i, facility in m.nodes.facilities.items():
        assert facility.category in m.nodes.categories

    # add_facility works
    number_of_items = m.nodes.number_of_items
    hospital = Hospital(
        h_category="LARGE",
        name="TEST FACILITY",
        n_beds=5,
        n_ventilator_beds=2,
        n_icu_beds=3,
        initial_icu=1,
        initial_normal=1,
        initial_normal_covid=0,
        initial_icu_covid=0,
        county=3,
    )
    m.nodes.add_facility(hospital)
    assert m.nodes.number_of_items == number_of_items + 1


def test_nh_ls_init(model):
    """Test that the nursing homes and LTACHS  with at least 10 scaled beds are initialized to close to 70% full
    Test that most of the people in them are from nearby counties
    """
    multiplier = get_multiplier(model.params)
    nh_probs = di.facility_to_county_probabilities("nh")
    lt_probs = di.facility_to_county_probabilities("ltach")
    # get the information for each nursing home and LTACH
    for i in model.nodes.category_ints["NH"] + model.nodes.category_ints["LT"]:
        facility = model.nodes.facilities[i]

        beds = max(1, int(round((facility.beds * multiplier))))
        if beds > 10:
            beds_initialized = max(1, int(round(len(model.nodes.facilities[i].agents))))
            # TODO assert (beds_initialized / beds) < 0.85 and (beds_initialized / beds) > 0.55

            if facility.category == "NH":
                counties = nh_probs[facility.name][1]
            else:
                counties = lt_probs[facility.name][1]

            nearby = 0
            far_away = 0
            for unique_id in model.nodes.facilities[i].agents:
                if model.county_codes[unique_id] in counties[:50]:
                    nearby += 1
                else:
                    far_away += 1
            # TODO assert far_away <= nearby


def test_select_los(model):
    m = deepcopy(model)

    # Move to different locations, making sure the LOS is within reasonable values
    for i in range(10):
        m.movement.current_los[0] = 0
        # Hospitals
        for category in ["UNC", "LARGE", "SMALL"]:
            for h_int in m.nodes.category_ints[category]:
                m.movement.current_los[0] = 0
                m.movement.assign_los(unique_id=0, new_location=h_int)
                assert 0 < m.movement.current_los[0] < 100
        # NH
        for nh_int in m.nodes.category_ints["NH"]:
            m.movement.current_los[0] = 0
            m.movement.assign_los(unique_id=0, new_location=nh_int)
            assert 0 < m.movement.current_los[0] < 2001

    # Providing a non-int should error
    with pytest.raises(KeyError):
        m.movement.assign_los(0, "UNC CH HILL")
    with pytest.raises(KeyError):
        m.movement.assign_los(0, 1.1)


def test_find_location_transitions(model):
    county = model.county_codes[0]
    age = model.age_groups[0]
    transitions = model.movement.find_location_transitions(county, age, model.movement.location.values[0])
    # Enough transitions to equal categories
    assert len(transitions) == len(model.nodes.categories)

    # All possible locations work
    for a_loc, facility in model.nodes.facilities.items():
        assert sum(model.movement.find_location_transitions(135, 2, a_loc)) > 0


def test_community_movement(model):
    m = deepcopy(model)
    # Make first 1k people move
    m.movement.location.probabilities[0:10].fill(1)
    m.movement.location.values.fill(0)
    # No one should be in the community anymore
    m.movement.community_movement()
    # Most people should move (if pop is limited, hospital bed counts may be to low to allow this)
    assert sum(m.movement.location.values[0:10] != 0) > 8


@pytest.mark.skip("TODO")
def test_facility_movement(model):
    """These are tests that are a part of the calibration run and are checked there."""
    # Cannot pickle SQL connection
    m = deepcopy(model)

    h_int = m.nodes.category_ints["UNC"][0]

    # Find people at this facility
    unique_ids = m.unique_ids[m.movement.location.values == h_int]

    m.time = 1
    for unique_id in unique_ids:
        m.movement.current_los[unique_id] = 1
        m.movement.leave_facility_day[unique_id] = m.time
    # Run Function
    m.movement.facility_movement()

    # Everyone should move
    new_locations = m.movement.location.values[unique_ids]
    assert all(new_locations != h_int)

    # Most people should go to the community
    if new_locations > 10:
        assert sum(new_locations == 0) / len(new_locations) > 0.6


@pytest.mark.skip("TODO")
def test_readmission_movement(model):
    # Cannot pickle SQL connection
    m = deepcopy(model)

    # Make sure everyone is at home with a readmission ready to occur
    m.movement.current_los = dict()
    m.movement.leave_facility_day = dict()
    COMMUNITY_id = 0
    UNC_0_id, UNC_0 = next((i, f) for i, f in m.nodes.facilities.items() if f.id.startswith("UNC"))

    for unique_id in m.unique_ids:
        m.movement.location.values[unique_id] = COMMUNITY_id
        m.movement.readmission_date[unique_id] = m.time
        m.movement.readmission_location[unique_id] = UNC_0_id

    m.movement.readmission_movement()

    # Everyone should move
    # TODO: convert_events_table_to_df doesn't seem to generate any events
    # len(e) == 0
    e = m.convert_events_table_to_df()
    assert len(e) == len(m.population)

    # Everyone should go to UNC
    assert len(e.New.unique()) == 1
    assert e.New.unique()[0] == UNC_0_id

    # Everyone should be at a location
    assert sum(m.movement.location.values != 0) == len(m.population)
    assert len(m.movement.current_los) == len(m.population)

    # No one should be readmitable (values are removed during the next time step)
    m.time = m.time + 1
    m.movement.readmission_movement()
    assert len(m.movement.readmission_location) == 0
    assert len(m.movement.readmission_date) == 0


def test_initialization(model):
    # This could use expansion
    # Unique ID
    assert model.unique_ids[0] == 0
    assert model.unique_ids[-1] == len(model.population) - 1

    # There are only 3 age groups, 0, 1 and 2
    assert 0 <= min(model.age_groups)
    assert 2 >= max(model.age_groups)

    # Community Probability should be small
    assert 0 <= model.movement.location.probabilities.max() < 0.05
    assert 0 <= model.movement.location.probabilities.min() < 0.05

    # Agents should be alive
    assert all(model.life.values == LifeState.ALIVE.value)


# TODO
# test_select_st_hospital
# update_location_transitions()
# select_location()

__all__ = ["model"]
