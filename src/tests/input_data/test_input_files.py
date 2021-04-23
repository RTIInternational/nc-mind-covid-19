import src.data_input as di


def test_county_hospital_distances():
    """Test the county hospital distance file to make sure each county and each hospital are there
    make sure all distances are non-negative
    """
    hospitals = di.hospitals()
    ch = di.county_hospital_distances()
    hospital_names = hospitals.Name.values

    for county in range(1, 201, 2):
        assert county in ch.keys()
        assert len(ch[county]) >= len(hospital_names)


def test_length_of_stay_files():
    nh1 = di.nh_los()
    nh2 = di.nh_los2()
    """Test the cLOS_MDS file and the nh_time_until_leaving file
    All values should be non-negative
    NOTE: the second one should have future tests regarding how it is created.
    A snapshot of time left after 100 days
    """
    for los in [nh1, nh2]:
        for col in los.columns:
            vals = los[col].dropna().astype(int).tolist()
            assert all([v >= 0 for v in vals])


def test_id_files():
    """Test nh_ids.csv, lt_ids.csv, hospital_ids.csv"""
    hs = di.hospitals()
    nh = di.nursing_homes()
    lt = di.ltachs()

    # for the hospital file, check capacities
    assert all(hs["Acute_NonCovid_Agents"] >= 0) and all(hs["Acute_NonCovid_Agents"] <= 10000)
    assert all(hs["ICU_NonCovid_Agents"] >= 0) and all(hs["ICU_NonCovid_Agents"] <= 10000)
    assert all(hs["Acute_Covid_Agents"] >= 0) and all(hs["Acute_Covid_Agents"] <= 10000)
    assert all(hs["ICU_Covid_Agents"] >= 0) and all(hs["ICU_Covid_Agents"] <= 10000)

    # for all files, check
    for id_file, cat in zip([hs, nh, lt], [["UNC", "LARGE", "SMALL"], ["NH"], ["LT"]]):

        # category has correct value(s)
        assert set(id_file["Category"].unique()) == set(cat)
        if cat[0] in ["NH", "LT"]:
            assert all(id_file["Beds"] > 0)

    assert all(hs["ICU Beds"] >= 0)
    assert all(hs["Total Beds"] >= hs["ICU Beds"])
    assert all(hs["Total Beds"] >= hs["Acute Beds"])
