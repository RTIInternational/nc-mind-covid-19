## Simple SEIR Modeling
This directory contains code to create COVID-19 projections for each NC county.

Originally, running the SEIR model occurred outside of the ABM. This was causing some issues and required users to 
make sure that the SEIR model was ran before starting an ldm ABM run.

For now, the SEIR model will be ran when an ldm model is initiated. Users do not need to run any additional code.

### Grabbing The Data
Data can be obtained two ways: stakeholder (limited access) or through the News and Observer.

#### Option 1: From Stakeholder
If obtained from stakeholder, rename the file to `RTI_CasesbyCounty.xlsx` and save in this directory. 

Stakeholder data must be reshaped from the format of `RTI_CasesbyCounty.xlsx` to the format expected for `cases_by_day.csv`. You can do this one of two ways.

1. Manually run `python seir/src/cases_by_day_convert.py`, or
2. Add the `-s` flag when you start the production run using `run_all.sh`, e.g. `run_all.sh -s`. This adds an additional step to the production run to automatically run `cases_by_day_convert.py`.

#### Option 2: From N&O
The cases by data is found [here](https://www.newsobserver.com/news/local/article241168731.html)

Look for the graph "TIMELINE OF NUMBER OF NEW CORONAVIRUS CASES BY COUNTY BY DAY". There is a link under the graph that says "Get the data".

Download the file and:

* remove all rows with null dates
* only keep four columns: County, Date, New Case, Total Cases
* save the file as seir/cases\_by\_day.csv

**The default `cases_by_day.csv` file in this directory is from the News and Observer**
