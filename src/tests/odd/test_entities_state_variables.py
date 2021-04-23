from src.tests.fixtures import model_function_scoped, model
import pytest
import src.data_input as di

ldm = model
ldm_copy = model_function_scoped
hospitals = di.hospitals()


class TestLocationNodeCounts:
    """These are from the Location type table and counts"""

    def test_community_count(self, ldm):
        """1 Community Node"""

        community_nodes = [f for f in ldm.nodes.facilities if ldm.nodes.facilities[f].category == "COMMUNITY"]
        assert len(community_nodes) == 1

    def test_unc_stach_count(self, ldm):
        """10 UNC Health Care (UNC) STACHs"""

        unc_nodes = [n for i, n in ldm.nodes.facilities.items() if n.category == "UNC"]
        assert len(unc_nodes) == len(hospitals[hospitals.Category == "UNC"])

    def test_large_stach_count(self, ldm):
        """non-UNC hospitals with ≥400 beds"""

        large_stach_nodes = [n for i, n in ldm.nodes.facilities.items() if n.category == "LARGE"]
        assert len(large_stach_nodes) == len(hospitals[hospitals.Category == "LARGE"])

    def test_small_stach_count(self, ldm):
        """non-UNC hospitals with <400 beds"""

        small_stach_nodes = [n for i, n in ldm.nodes.facilities.items() if n.category == "SMALL"]
        assert len(small_stach_nodes) == len(hospitals[hospitals.Category == "SMALL"])

    def test_lt_count(self, ldm):
        """non-UNC hospitals with <400 beds"""

        lt_nodes = [n for i, n in ldm.nodes.facilities.items() if n.category == "LT"]
        assert len(lt_nodes) == len(di.ltachs())

    def test_nh_count(self, ldm):
        """non-UNC hospitals with <400 beds"""

        nh_nodes = [n for i, n in ldm.nodes.facilities.items() if n.category == "NH"]
        assert len(nh_nodes) == len(di.nursing_homes())


class TestFacilityAttributes:
    def test_facility_attrs(self, ldm):
        """Test existence of attributes in table"""
        # coordinates?

        hospitals = [n for i, n in ldm.nodes.facilities.items() if n.category in ldm.nodes.hospital_categories]
        attrs = ["n_total_beds", "name", "category", "county", "agents"]
        for hospital in hospitals:
            for attr in attrs:
                assert hasattr(hospital, attr)

    def test_facility_categories(self, ldm):
        """Categories are defined in table"""

        categories = ["UNC", "SMALL", "LARGE", "NH", "LT", "COMMUNITY"]
        nodes = [n for i, n in ldm.nodes.facilities.items()]
        for node in nodes:
            assert node.category in categories

    def test_facility_counties(self, ldm):
        """
        Not all counties have a facility.

        Last check on 8/4/20 indiciates 81 counties with a facility
        """
        hospitals = [n for i, n in ldm.nodes.facilities.items() if n.category in ldm.nodes.hospital_categories]
        assert len(set(n.county for n in hospitals)) > 0

    def test_facility_ids(self, ldm):
        nodes_int_ids = [i for i, n in ldm.nodes.facilities.items()]
        assert min((n for n in nodes_int_ids)) == 0
        assert max((n for n in nodes_int_ids)) == len(ldm.nodes.facilities) - 1


class TestAgentStates:
    """Agents have three main state variables: location, life, and COVID-19 status."""

    def test_location_state(self, ldm):
        assert hasattr(ldm, "movement")  # movement is a NorthCarolina object
        assert len(ldm.movement.location.values) == len(ldm.population)

    def test_life_state(self, ldm):
        assert hasattr(ldm, "life")
        assert len(ldm.life.values) == len(ldm.population)

    def test_covid_state(self, ldm):
        assert hasattr(ldm, "disease")
        assert len(ldm.disease.covid19.values) == len(ldm.population)


class TestAgentDemographics:
    """Agent State variables: demographics"""

    def test_age(self, ldm):
        assert hasattr(ldm, "age_groups")
        assert len(ldm.age_groups) == len(ldm.population)

    def test_county(self, ldm):
        assert hasattr(ldm, "county_codes")
        assert len(ldm.county_codes) == len(ldm.population)
        assert len(set(ldm.county_codes)) == 100

    def test_concurrent_conditions(self, ldm):
        assert hasattr(ldm, "concurrent_conditions")
        assert len(ldm.concurrent_conditions) == len(ldm.population)


class TestAgentTempStates:
    """Throughout the model run, agents who transition locations receive additional state variables."""

    def test_los(self, ldm_copy):
        ldm = ldm_copy
        assert hasattr(ldm.movement, "current_los")
        ldm.step()
        # bool({}) returns False, this is asserting the dictionary is not empty
        assert bool(ldm.movement.current_los), "current_los empty after 1 step"

    def test_leave_facility_day(self, ldm_copy):
        ldm = ldm_copy
        assert hasattr(ldm.movement, "leave_facility_day")
        ldm.step()
        assert bool(ldm.movement.leave_facility_day), "leave_facility_day empty after 1 step"

    def test_previous_location(self, ldm_copy):
        ldm = ldm_copy
        assert hasattr(ldm.movement, "leave_facility_day")
        ldm.step()
        assert len(set(ldm.movement.location.previous)) > 1, "location.previous unchanged after 1 step"

    def test_readmission_date(self, ldm_copy):
        ldm = ldm_copy
        assert hasattr(ldm.movement, "readmission_date")
        for day in range(10):
            ldm.time = day
            ldm.step()  # takes time for readmission
        assert bool(ldm.movement.readmission_date), "readmission_date empty after 10 steps"

    def test_readmission_location(self, ldm_copy):
        ldm = ldm_copy
        assert hasattr(ldm.movement, "readmission_location")
        for day in range(10):
            ldm.time = day
            ldm.step()  # takes time for readmission
        assert bool(ldm.movement.readmission_location), "readmission_location empty after 10 steps"

    def test_icu_status(self, ldm_copy):
        ldm = ldm_copy
        assert hasattr(ldm, "icu_status")
        ldm.step()
        assert len(set(ldm.icu_status)) > 1, "icu_status unchanged after 1 step"

    def test_covid_testing(self, ldm_copy):
        ldm = ldm_copy
        assert hasattr(ldm.disease, "covid19tested")
        ldm.time = 1
        ldm.step()
        assert len(set(ldm.disease.covid19tested.values)) > 1, "covid19tested has only one unique value after 51 steps"
