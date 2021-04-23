## Required Steps for a Model Run

Before running a model, you must run:

```
python model_runs/src/create_hospital_bed_counts.py
python model_runs/src/prepare_transitions.py --data_file="<path_to_additional_data_data>.csv"
```

We now give a quick description of these scripts.

### Create hospital bed counts:

By default our repo uses hospital bed counts from 2017. However, if updated bed counts are available, the model can use these updated counts. 

```
python model_runs/src/update_beds_counts.py
```

### Create Transition Files

The transition probabilities are calculated with stakeholder data on a weekly basis, however this data is protected and 
can only be run by someone with access to this data. The default transition probabilities are the most recent 
probabilities at the time of publishing. The transition files can also be generated from SHEPS Center data.

To create default transition files with SHEPS data, run:

```
python model_runs/src/prepare_transitions.py
```

To create transition files based on stakeholder provided data, run:

```
python model_runs/src/prepare_transitions.py --data_file="<path_to_additional_data_data>.csv"
```


## Steps Completed Previously

### SHEPs PDF Data

Transition probabilities rely on public data from the SHEPS Center, which are saved as PDF reports. Although this has 
already been completed previously, to create a cleaned CSV from the provided PDFs, run:

```
python model_runs/src/clean_pdf_csvs.py
```

###  Death Probabilities:
In `data/` there is an Excel file called `mortality.xlsx`. These mortality rates were taken from CDC
Wonder. We looked at NC deaths by age, sex, and race. [link](https://wonder.cdc.gov/controller/datarequest/D140;jsessionid=5A767ED9BA64A7E66597304A920BC503)

We use death rates by age, with multipliers for risk based on location. Death probabilities are in the 
`parameters.json` file. 


### Synthetic Population

The RTI Synthetic Population is required to run this model. To simplify and prepare the synthetic population, we ran the following:

```
python model_runs/src/extract_syn_pop.py
```

