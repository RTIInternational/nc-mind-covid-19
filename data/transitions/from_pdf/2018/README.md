# Convert PDFs to CSVs for HAI project

The necessary data for the CDC HAI project is exported in PDFs that contain both tables and text. Since there are hundreds of these tables, a script is necessary to extract the data. Follow these steps to replicate the process for 2018. The column names will need to be updated with future years.

## Step 1 Get PDFs

Download the two relevant pdfs for CDC HAI. [1](https://www.shepscenter.unc.edu/wp-content/uploads/2020/05/ptchar_all_and_by_hosp_2018_and.pdf) [2](https://www.shepscenter.unc.edu/wp-content/uploads/2020/05/ptorg_pt_res_by_hosp_2018.pdf)

Move both files to here (this directory).

## Step 2 Initial Processing of Patient Characteristics by Hospital (ptchar)

Make sure you install: 

```
pip install camelot-py
pipt install pdfquery
```

After setting up your environment for this repo, run the following from this directory:

`python 1_read_ptchar_pdf.py`

This will create tables for all hospitals with all years reported. It will also create a file `<year>_manual_evaluation_ptchar.txt` with any hospitals that did not report for all years.

## Step 3 Manually Evaluate Hospitals and Add As Needed to Processing File

For all hospitals in `<year>_master_evaluation_ptchar.txt`, manually evaluate whether they reported data in the most recent year. If they did, note the page numbers that were not processed and add these page numbers to the list of pages in

`2_add_outlier_hospitals.py`.

## Step 4 Final Processing of ptchar

After adding the page numbers in step 3, run the following:

`python 2_add_outlier_hospitals.py`

This will create `master_ptchar_final.py` which includes the output of all tables.

### Note on page 342 (from 2017)

Some pages simply fail. For example, page 342 from 2017 did not read correctly in the PDF process. It is not clear why this is. The data from this page was manually added to `master_ptchar_final.csv`.

## Step 5 Create CSV subset for CDC HAI

CDC HAI only uses a subset of the variables in the tables. To capture this subset, after step 4, run the following:

`python 3_make_ptchar_subset.py`

This will `create subset_ptchar_for_analysis.csv`. It will also create a total column and check for differences between expected and calculated output.

## Step 6 Processing of Patient Residence by Hospital

Process the ptorg file with the following:

`python 4_read_ptorg_pdf.py`

Note that the Actual total and Calculated total do not always match as the PDF values do always add up to 100%. The difference is made into a new row called Unreported.


