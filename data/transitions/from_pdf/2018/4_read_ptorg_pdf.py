# Read ptchar pdfs

import pandas as pd
import numpy as np
import camelot
import pdfquery

from helpers import ptchar_var_map


# specify file and page range here
file = "ptorg_pt_res_by_hosp_2018.pdf"
year = "2018"
page_range = range(1, 115)

page_list = list(page_range)

# query pdf for hospital titles
pdf = pdfquery.PDFQuery(file)

master_df = pd.DataFrame()

# for each table (assuming 1 per page) determine the hospital and extract necessary data
for i in range(len(page_list)):

    # read pdf for tables
    table = camelot.read_pdf(file, flavor="lattice", pages=str(page_list[i]), process_background=True)

    # extract the hospital name for that page
    pdf.load(page_list[i] - 1)
    label = pdf.pq('LTTextLineHorizontal:contains("Patient County of Residence by Hospital")')
    left_corner = float(label.attr("x0"))
    bottom_corner = float(label.attr("y0"))
    right_corner = float(label.attr("x1"))
    top_corner = float(label.attr("y1"))
    hospital = pdf.pq(
        'LTTextLineHorizontal:in_bbox("%s, %s, %s, %s")' % (left_corner, bottom_corner, right_corner, top_corner)
    ).text()
    hospital_string = hospital.replace(" ", "_").lower().splitlines()[0][42:]

    # extract the dataframe for that page
    df = table[0].df

    # replace new line symbols with spaces
    df.replace({"\n": " "}, regex=True, inplace=True)

    # set first row to column headers
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])

    print("processing " + hospital_string)

    df["hospital"] = hospital_string

    mini_df = df[["RESIDENCE", "TOTAL CASES", "hospital"]]

    master_df = pd.concat([master_df, mini_df], axis=0, sort=False)

final = pd.DataFrame()
final = master_df.pivot_table(
    index="RESIDENCE", columns="hospital", values="TOTAL CASES", aggfunc=lambda x: " ".join(x)
)
final.loc["Actual", :] = final.loc["HOSPITAL TOTAL", :]
final = final.drop(["HOSPITAL TOTAL"])
final = final.replace({",": ""}, regex=True).apply(pd.to_numeric, 1)
final = final.apply(pd.to_numeric)
final.loc["Calculated", :] = final.iloc[:-1, :].sum(axis=0, skipna=True)
final.loc["Unreported", :] = final.loc["Actual", :] - final.loc["Calculated", :]

final.columns = [item.split("_-_")[1] for item in final.columns]
final.columns = [item.replace("_", " ").title() for item in final.columns]

export_file = year + "master_ptorg_final.csv"
final.to_csv(export_file)
