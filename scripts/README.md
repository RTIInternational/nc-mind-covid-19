# Scripts

## Parallel Processing

Each run of the model takes anywhere from a few minutes with 1 million agents to nearly half an hour with 10 million
agents. We set up a parallel process to independently and simultaneously perform model runs.

Parallel processing makes use of [GNU Parallel](https://www.gnu.org/software/parallel/) to create an ad-hoc compute
cluster.

### Prerequisite / Setup


These steps generally only need to be completed once:

1. On your target servers:
    * Install Docker and docker-compose if not already installed
    * If using an Ubuntu based server for the first time, run the contents of `server_init.sh` in the
    initialization routine during creation or afterward over an SSH session to prepare it as a compute node
2. Setup [passwordless SSH](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2) to all of your target compute nodes
3. Install GNU Parallel (macOS)
```
brew install parallel
```
4. Ensure that you have the synthetic population saved as a parquet file as described in the [data](../data) directory
5. Ensure that you have created at least the default transition probability files as described in [model_runs](../model_runs/README.md) if they are not already present in the repo.

### Directions for a Production-Level Run
1. Define the list of servers you want to use for your cluster in `servers` dictionary in [create_runs.py](../model_runs/src/create_runs.py)
    * ":" is a special value for GNU parallel SSH logins which means "local machine" (i.e., no SSH connection).  
    Models run on this "host" will run on your computer. The default is this option with two CPUs.
    * Data will be copied to `/tmp/covid19` on any server used for model runs (including the local machine, optionally).
2. Follow the instructions in [config_files](../config_files) to set up the config files for the run. The default
config directory is [test_parallelization](../config_files/test_parallelization).
3. Follow applicable directions in [seir](../seir)
4. Modify the config dir [../model_runs/src/create_runs.py](../model_runs/src/create_runs.py) with the config directory created in step 2
5. Run `./scripts/run_all.sh` from root in the virtualenv will execute all of the subordinate scripts:
	* `make_nodes.sh`: generates the `nodes.txt` that `parallel` uses
	* `clean.sh`: cleans any existing directories on the target servers
	* `create.sh`: creates all of the runs and parameter files for the configs
	* `deploy.sh`: builds the docker containers on target servers
	* `copy_directories.sh`: copies run directories to the correct servers
	* `run.sh`: runs all directories on each server
	* `gather.sh`: gathers the results back to the server this is run from and store them locally
	* `aggregate.sh`:
		* aggregates all runs from the same config file and generate output CSVs and plots
		* aggregate across all iterations from all config files and generates output CSV
	* `clean_capacities.sh`: removes the hospital capacity json from servers if it exists (this file has strict sharing rules)

	**Note**: Run `./scripts/run_all.sh -s` with the **-s** flag to also run `seir.sh` prior to the other subordinate scripts. This script prepares case data file used as input for SEIR model. It should not be used if `cases_by_day.csv` was manually updated, e.g. with Option 2 in Step 3.

	**Note**: Run `time ./scripts/run_all.sh` if you are also interested in the time it takes to run the output

7. Check [here](../parallel_reports) for the output from the run, including relevant CSV and PNG files. Generally the
 main file of interest is `smoothed_master_results.csv`

## Troubleshooting

* If you are having "cannot find module" issues, you may need to set the PYTHONPATH environment variable, which you can set to add additional directories where python will look for modules and packages. You can resolve this in two ways:

1. If using vscode, add the following lines to `./.vscode/settings.json` and save:
	```
	"terminal.integrated.env.<os>": {
        "PYTHONPATH": "../covid19:$PYTHONPATH"
    }
	```
	(<os> takes the value of your operating system: osx, windows, etc.)

2. Alternatively, run the following line from root in the virtualenv each time prior to running `./scripts/run_all.sh`:
	```
	export PYTHONPATH=../covid19:$PYTHONPATH
	```
* If your script is hanging on the `deploy.sh` step, you might have a problem with the line `parallel --sshloginfile scripts/nodes.txt --onall ::: "mkdir -p $RUN_PATH"`. Try checking that the folders `/tmp/<username>/covid19` exist on each server, and make them manually if they don't. If you know the folders exist, and the script is still hanging, try commenting that line out in `deploy.sh`.
* If you get the following error or something similar in the `copy_directories.sh` step `rsynch: recv_generator: mkdir "/tmp/<username>/covid19/model_runs/parallel_runs/run_#" failed: Permission denied (13)`, there might be a permission problem with creating the folders. Check the ownership of the folders with `ls -al <path>/parallel_runs`. If it is `root`, then someone with sudo access will need to change the ownership of `/tmp/<username>` to your username. This should resolve the issue.
