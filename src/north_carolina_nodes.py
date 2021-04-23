from typing import Dict, Union

import src.data_input as di
from src.facilities import Community, Hospital, LongTermCareFacility, NursingHome

FacilityTypes = Union[Community, Hospital, NursingHome, LongTermCareFacility]


class NcNodeCollection:
    """A class for holding all location nodes for North Carolina"""

    def __init__(self, multiplier: float):

        self.community = 0
        self.number_of_items = 0
        self.facilities: Dict[int, FacilityTypes] = dict()
        self.category_ints: Dict[int, str] = dict()
        self.name_to_int: Dict[str, int] = dict()
        self.categories = ["COMMUNITY", "UNC", "LARGE", "SMALL", "LT", "NH"]
        self.hospital_categories = ["UNC", "LARGE", "SMALL"]
        # Add Community
        self.add_facility(facility=Community())
        # Add Hospitals
        hospitals = di.hospitals()
        for _, row in hospitals.iterrows():
            ventilator_beds = int(round(0.95 * row["ICU Beds"]))
            hospital = Hospital(
                name=row["Name"],
                county=row.County,
                h_category=row.Category,
                n_beds=int(row["Total Beds"]),
                n_icu_beds=int(row["ICU Beds"]),
                n_ventilator_beds=ventilator_beds,
                initial_normal=row["Acute_NonCovid_Agents"],
                initial_icu=row["ICU_NonCovid_Agents"],
                initial_normal_covid=row["Acute_Covid_Agents"],
                initial_icu_covid=row["ICU_Covid_Agents"],
            )
            hospital.apply_bed_multiplier(multiplier)
            self.add_facility(hospital)
        # Add Nursing Homes
        nhs = di.nursing_homes()
        for row in nhs.itertuples():
            nh = NursingHome(beds=row.Beds, name=row.Name, model_int=self.number_of_items)
            self.add_facility(nh)
        # Add LTACHs
        lt_ids = di.ltachs()
        for row in lt_ids.itertuples():
            ltach = LongTermCareFacility(beds=row.Beds, name=row.Name, model_int=self.number_of_items)
            self.add_facility(ltach)

        self.all_hospitals = set(self.category_ints["UNC"] + self.category_ints["LARGE"] + self.category_ints["SMALL"])
        self.all_hospital_names = [self.facilities[item].name for item in self.all_hospitals]
        self.large_unc = [item for item in self.category_ints["UNC"] if self.facilities[item].base_n_total_beds > 400]

    def __repr__(self):
        repr_values = [
            "NcNodeCollection",
            f"UNC Hospitals: {self.n_per_category('UNC')}",
            f"LARGE Hospitals: {self.n_per_category('LARGE')}",
            f"SMALL Hospitals: {self.n_per_category('SMALL')}",
            f"Nursing Home: {self.n_per_category('NH')}",
            f"LTACHs: {self.n_per_category('LT')}",
        ]
        return " - ".join(repr_values)

    def n_per_category(self, category: str) -> int:
        return len(self.category_ints[category])

    def add_facility(self, facility):
        category = facility.category
        facility.model_int = self.number_of_items
        self.facilities[facility.model_int] = facility
        self.category_ints[category] = self.category_ints.get(category, []) + [facility.model_int]
        self.name_to_int[facility.name] = facility.model_int
        self.number_of_items += 1
