#!/bin/bash
# -x: Trace execution of nested scripts and commands
# -e: Exit on error
# -u: Error on accessing unset variables
set -xeu

function seir {
    ./scripts/seir.sh
}

function run_model {
    ./scripts/make_nodes.sh
    ./scripts/create.sh
    ./scripts/deploy.sh
    ./scripts/clean.sh
    ./scripts/copy_directories.sh
    ./scripts/run.sh
    ./scripts/gather.sh
    ./scripts/aggregate.sh
    ./scripts/clean_capacities.sh
}

###

while getopts ":s" flag
do
  case ${flag} in
    s ) echo "Running SEIR script"
        seir
  ;;
  esac
done

###

echo "Running Model" 
run_model