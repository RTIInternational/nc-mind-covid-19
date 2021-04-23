## Preparing configuration files

If a new configuration file is provided and it is for a production run, create a new directory in `config_files/` with
the date,
such as

```
config_files/04_22_2020/
```

If this is a draft or test run, config files are not tracked in Git and the directory can be created in in
[draft_run_configs](draft_run_configs). This directory is ignored by Git:

```bash
config_files/draft_run_configs/<m_d_Y>
```

Copy the configuration file (the provided [template](configs.xlsx) to the directory created above
and fill out all cells.

The definitions are:

* **Starting Re range min:** the minimum Re bound to be sampled
* **Starting Re range max:** the maximum Re bound to be sampled
* **# runs:** the number of runs per scenario; use 1 for testing locally, use 100 or more for production
* **# agents:** number of agents to use, default 2.5 million
* **Reported Cases Multiplier (for total infections):** multiplier to use on reported cases to estimate number of
unreported infections; default 10
* **Days to run:** number of days for the COVID model to run. Model will start based on the last date of available data

Run with argument as directory of location for configs:

```
docker-compose run --rm covid bash -c "python3.6 config_files/src/make_configs.py 04_22_2020"
```

In [create_runs.py](../model_runs/src/create_runs.py), specify the directory name where the configs
files are located for `config_dir`
