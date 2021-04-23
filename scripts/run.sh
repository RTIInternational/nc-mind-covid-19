#!/bin/bash

set -eu

source ./scripts/env.sh

parallel --eta --progress --resume-failed --joblog ./log --sshloginfile scripts/nodes.txt --nonall "cd $RUN_PATH; docker-compose run --rm covid bash -c 'python3.6 model_runs/src/parallel_runs.py'"
