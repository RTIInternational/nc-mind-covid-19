from geopy import distance
from pathlib import Path
import json
import pandas as pd
import argparse as arg

if __name__ == "__main__":
    description = """Adds lat/lon values to id file based on a seperate address file.
    Make sure before running this script that you have set the LOCATIONIQ_KEY environment variable!"""

    parser = arg.ArgumentParser(description=description)

    parser.add_argument(
        "--id_file", default="geocoderesults.xlsx", help="The path to a file with location IDs as well as Lat/Lons."
    )
    parser.add_argument("--loc_type", default="hospital", help="The type of location we are getting the distances for.")

    args = parser.parse_args()
    id_path = args.id_file
    loc_type = args.loc_type

    county_centroids = json.loads(Path("county_centroids.json").read_text())

    county_ids = pd.read_csv("../county_codes.csv", usecols=["County", "County_Code"]).assign(
        County=lambda df: df["County"].str.upper()
    )

    county_id_map = {row.County: row.County_Code for row in county_ids.itertuples()}

    if id_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(id_path)
    else:
        df = pd.read_csv(id_path)

    hospital_locations = df.rename(columns={"LAT": "lat", "LON": "lon"}).dropna(subset=["lat", "lon"])

    hospital_data = [(row["Name"], row.lat, row.lon) for inde, row in hospital_locations.iterrows()]

    centroid_distances = {}

    for county, data in county_centroids.items():
        county_id = county_id_map[county]
        county_lat_lon = (data[0], data[1])
        county_distances = []
        for hospital, lat, lon in hospital_data:
            hospital_lat_lon = (lat, lon)
            dist = distance.distance(county_lat_lon, hospital_lat_lon).miles
            county_distances.append({"Name": hospital, "distance_mi": dist})
        centroid_distances[county_id] = county_distances

    for county, distances in centroid_distances.items():
        centroid_distances[county] = list(sorted(distances, key=lambda x: x["distance_mi"]))
    Path(f"county_{loc_type}_distances_sorted.json").write_text(json.dumps(centroid_distances, indent=4))
