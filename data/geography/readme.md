## Hospital Geocoding

NOTE: these scripts expect that you are running them from the geography directory.

1. `geocode.py` uses LocationIQ to search for Lat/Lon values by hospital name.
  * Arguments:
    * `--address_file`: The path to a file with either location names or addresses. At minimum should have ID and NAME fields.
    * `--sheet_name`: If the address file is an excel file, this is the name of the sheet to use.
    * `--id_file`: The path to a file with location IDs you want to add lat/lons to.
    * `--loc_type`: The type of facility. This is used to subset if the address file has multiple types.
    * `--address_cols`:A list of column names to use for the address, seperated by commas.
2. `format_geocode.py` takes the LocationIQ data and validates it. Geocoding returns a list of results per location, so we check if the first result has a lat/lon and is a "hospital" type. If so, keep it, if not, assume it was not the correct geocoding. Export this file as `geocoderesults.csv`
3. `geocoderesults.xlsx` was used to manually search using Bing maps and find lat/lon for remaining locations. It has fields noting if there were name ambiguities or other issues with locating the hospitals online.
4. `calculate_distance.py` calculates the distance from each of the county centroids to each facility. It outputs `county_<facility type>_distances.json` and `county_<facility type>_distances_sorted.json`, where the latter has the distance from county centroids to facilities sorted by closest to furthest.
  * Arguments:
    * `--id_file`: The path to a file with location IDs as well as Lat/Lons.
    * `--loc_type`: The type of facility we are using.

### Run the following:

To calculate the distances for each category, run the following:

```
python calculate_distance.py --id_file=../IDs/hospitals.csv --loc_type=hospital
python calculate_distance.py --id_file=../IDs/ltachs.csv --loc_type=ltach
python calculate_distance.py --id_file=../IDs/nursing_homes.csv --loc_type=nh
```


## Example use

```python
from pathlib import Path
import json   

county_to_hospital_distances = json.loads(Path("county_hospital_distances_sorted.json").read_text())

# get all hospitals under 100 miles for ALAMANCE (County_Code=1) county

county_code = "1"
alamance_100_miles = [hospital for hospital in county_to_hospital_distances[county_code] if hospital['distance_mi'] < 100]

# Get nearest to county centroid
closest = alamance_100_miles[0]

print(closest)
```
