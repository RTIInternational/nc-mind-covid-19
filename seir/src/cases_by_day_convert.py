import pandas as pd
import datetime as dt
from seir.src.simple_seir import prepare_data
import plotly.express as px


if __name__ == "__main__":
    df = pd.read_excel("seir/RTI_CasesbyCounty.xlsx")
    if list(df) == ["County", "FIPS", "reportdate", "Confirmed", "Probable"]:
        df.rename(columns={'reportdate': 'Date',
                           'Confirmed': 'New cases'}, inplace=True)
        df = df[['County', 'Date', 'New cases']]
    else:
        df.columns = ["County", "Date", "New cases"]
    df = df[~df.County.isna()]
    df = df[~df.Date.isna()]
    df = df[df.County != 'Missing'].copy()
    df = df[df.Date != "."].copy()
    df.Date = pd.to_datetime(df.Date, format="%d%b%Y")
    df = df[~(df.Date == pd.Timestamp(dt.datetime.today().date()))]
    df.sort_values(by=["County", "Date"], ascending=True, inplace=True)
    df = df[~df.County.str.contains("Frequency Missing")]
    df["County"] = df["County"].str.replace(" County", "")
    df["Total cases"] = df.groupby("County")["New cases"].transform(pd.Series.cumsum)
    df.to_csv("seir/cases_by_day.csv", index=False)

    res = prepare_data(include_flu_regions=True)[0]
    for key in res.keys():
        if key.startswith("LHD"):
            res[key].to_csv(f"seir/cases_by_day__{key}.csv")
