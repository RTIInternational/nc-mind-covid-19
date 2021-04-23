from dotenv import load_dotenv
import os
import requests
from pathlib import Path
import pandas as pd
from time import sleep
import pandas as pd
import json
import argparse as arg


load_dotenv()

API_KEY = os.environ.get("LOCATIONIQ_KEY")


def update_id_files_with_geo(df, output_file_path):
    """This function takes addresses, finds lat and lons, and adds
    them to an ID file
    """

    results = {}
    for index, row in df.iterrows():
        print(f"Getting info for {row.FACILITY}")
        r = requests.get(f"https://us1.locationiq.com/v1/search.php?key={API_KEY}&q={row.address}&format=json")
        if r.status_code == 200:
            result_json = r.json()
            results[row["PDF Name"]] = result_json
        else:
            print(f"Request Failed. {row['PDF Name']}, {row.FACILITY}. Address given: {row.address}")
            results[row["PDF Name"]] = {}
        sleep(1)

    one_res = {key: val[0] for key, val in results.items()}
    res_df = pd.DataFrame(one_res).transpose().reset_index()
    res_df = res_df[["index", "lat", "lon"]]
    res_df.columns = ["Facility_ID", "LAT", "LON"]

    id_df = pd.read_csv(output_file_path)
    both_df = id_df.merge(res_df, on="Facility_ID", how="left")
    both_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    description = """Adds lat/lon values to id file based on a seperate address file.
    Make sure before running this script that you have set the LOCATIONIQ_KEY environment variable!"""

    parser = arg.ArgumentParser(description=description)

    parser.add_argument(
        "--address_file",
        default="NC Licnesed Facilities STACH-LTACH-NH_3.27.18.xlsx",
        help="The path to a file with either location names or addresses."
        " At minimum should have ID and NAME fields.",
    )
    parser.add_argument(
        "--sheet_name",
        default="All Lic NC STACHs & LTACHs",
        help="If the address file is an excel file, this is the name of the sheet to use",
    )

    parser.add_argument(
        "--id_file",
        default="../IDs/lt_ids.csv",
        help="The path to a file with location IDs you want to add lat/lons to.",
    )

    parser.add_argument(
        "--loc_type",
        default="LTACH",
        help="The type of facility. This is used to subset if the address file has multiple types.",
    )

    parser.add_argument(
        "--address_cols",
        default="PHYS_ADDR1,CITY,STATE",
        help="A list of column names to use for the address, seperated by commas.",
    )

    args = parser.parse_args()

    id_path = args.id_file
    address_file = args.address_file
    sheet_name = args.sheet_name
    loc_type = args.loc_type

    address_cols = args.address_cols.split(",")

    if address_file.endswith((".xlsx", ".xls")):
        df = pd.read_excel(address_file, sheet_name=sheet_name)
    else:
        df = pd.read_csv(address_file)

    df["address"] = df.apply(lambda row: ", ".join([row[c] for c in address_cols]), axis=1)

    if "TYPE" in df.columns.values.tolist():
        df = df.loc[df["TYPE"] == loc_type].copy()

    update_id_files_with_geo(df, Path(id_path))
