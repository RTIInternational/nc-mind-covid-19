from pathlib import Path
import json
import pandas as pd

hospital_geocodes = json.loads(Path("geocoded_hospitals.json").read_text())

results = []
for k, v in hospital_geocodes.items():
    r = {"ID": k}
    try:
        first_item = v[0]
        item_hospital = first_item.get("type") == "hospital"
        if item_hospital:
            r["lat"] = first_item["lat"]
            r["lon"] = first_item["lon"]
    except KeyError:
        pass
    results.append(r)

pd.DataFrame(results).to_csv("geocoderesults.csv")
