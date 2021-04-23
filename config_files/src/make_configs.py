import pandas as pd
import argparse as arg
from datetime import datetime  # noqa
import dateutil.parser
from pathlib import Path
import src.data_input as di


if __name__ == "__main__":
    description = """ Create configuration files based on the provided config excel file."""

    parser = arg.ArgumentParser(description=description)
    parser.add_argument("directory", default="04.22.2020", help="The directory of the configuration file.")
    args = parser.parse_args()

    config_dir = Path("config_files", args.directory)

    df = pd.read_excel(config_dir.joinpath("configs.xlsx"))


    for _, row in df.iterrows():
        config = list()

        rmin = str(float(row["Starting Re range min"]))
        if rmin.split(".")[1] == "0":
            rmin_str = rmin.split(".")[0]
        else:
            rmin_str = rmin.split(".")[0] + "p" + rmin.split(".")[1]

        rmax = str(row["Starting Re range max"])
        if rmax.split(".")[1] == "0":
            rmax_str = rmax.split(".")[0]
        else:
            rmax_str = rmax.split(".")[0] + "p" + rmax.split(".")[1]

        runs_per = int(row["# runs"])

        num_agents = "{:,}".format(row["# agents"]).replace(",", "_")

        m = str(float(row["Reported Cases Multiplier (for total infections)"]))
        if m.split(".")[1] == "0":
            m_str = "x" + m.split(".")[0]
        else:
            m_str = "x" + m.split(".")[0] + "p" + m.split(".")[1]

        daystorun = row["Days to Run"]

        config.append("R0MIN={}".format(rmin) + "\n")
        config.append("R0MAX={}".format(row["Starting Re range max"] + 0.001) + "\n")
        config.append("RUNSPER={}".format(runs_per) + "\n")
        config.append("NUMAGENTS={}".format(num_agents) + "\n")
        config.append("MULTIPLIER={}".format(m) + "\n")
        config.append("DAYSINMODEL={}".format(daystorun) + "\n")

        out = "r0_" + rmin_str + "_" + rmax_str + "_" + m_str
        config.append("OUTPUTSTR='{}'".format(out))

        with open(config_dir.joinpath(out + ".config"), "w") as file:
            file.writelines(config)
