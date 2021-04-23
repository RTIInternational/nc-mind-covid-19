import pytest

from src.tests.fixtures import model
import src.data_input as di


def test_community():
    c = di.community_transitions()
    # 100 Counties, and 3 age groups
    assert c.shape[0] == 300

    # The sum of UNC, Large, Small, LT, and NH should be the probability
    assert max(c.Probability - c[["UNC", "LARGE", "SMALL", "LT", "NH"]].sum(axis=1)) < 1 * 10 ** -7

    # Agents cannot move to the community from the community
    value = c["COMMUNITY"].unique()
    assert len(value) == 1
    assert value[0] == 0

    # Only 65+ For nursing home
    assert max(c[c.Age_Group < 2].NH) == 0
    assert min(c[c.Age_Group == 2].NH) > 0


def test_county_discharge_files():
    discharges = di.county_discharges()
    hospitals = di.hospitals()

    # There are 100 counties
    assert len(discharges) == 100

    # Make sure all hospitals are represented
    items = [name for name in hospitals.Name if name in discharges.columns]
    assert len(items) == len(hospitals)


def test_nh(model):
    temp_df = di.facility_transitions()
    nh = temp_df[temp_df.Facility == "NH"]

    # Anyone less than age group 2 should not have any probabilities (i.e. community == 1)
    assert nh[nh.Age_Group < 2].COMMUNITY.mean() == 1

    # Rows should add to 1
    value = round(nh[list(model.nodes.categories)].sum(axis=1), 5).unique()
    assert len(value) == 1
    assert value[0] == 1

    # Community, LT, and NH all have specific values
    nh = nh[nh.Age_Group == 2]
    assert nh.COMMUNITY.mean() == pytest.approx(0.673, 0.1)
    assert nh.LT.mean() == 0
    assert nh.NH.mean() == 0


def test_lt(model):
    temp_df = di.facility_transitions()
    lt = temp_df[temp_df.Facility == "LT"]

    # All Ages can go to LT
    value = lt["Age_Group"].unique()
    assert len(value) == 3

    # Rows should add to 1
    value = round(lt[list(model.nodes.categories)].sum(axis=1), 5).unique()
    assert len(value) == 1
    assert value[0] == 1

    # LT can only be 0
    assert lt.LT.mean() == 0

    # Only 65+ For nursing home
    assert max(lt[lt.Age_Group < 2].NH) == 0
    assert min(lt[lt.Age_Group == 2].NH) > 0


def test_hospitals(model):
    hospitals = di.hospitals()
    transitions = di.facility_transitions()
    for category in ["LARGE", "SMALL", "UNC"]:
        hospital_names = hospitals[hospitals.Category == category].Name.values
        category_transitions = transitions[transitions.Facility.isin(hospital_names)]

        # Should be all facilities for that specific category
        values = category_transitions["Facility"].unique()
        for item in model.nodes.category_ints[category]:
            assert model.nodes.facilities[item].name in values

        # Rows should add to 1
        value = round(category_transitions[list(model.nodes.categories)].sum(axis=1), 5).unique()
        assert len(value) == 1
        assert value[0] == 1

        # Only 65+ For nursing home
        assert max(category_transitions[category_transitions.Age_Group < 2].NH) == 0
        assert min(category_transitions[category_transitions.Age_Group == 2].NH) > 0


__all__ = ["model"]
