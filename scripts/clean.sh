#!/bin/bash
set -eu
source ./scripts/env.sh
parallel --sshloginfile scripts/nodes.txt --onall ::: "cd $RUN_PATH && docker-compose run --rm covid rm -rf ./model_runs/parallel_runs/run_*"
parallel --sshloginfile scripts/nodes.txt --onall ::: 'running_container_ids=$(docker ps -q --filter ancestor=covid:19 --format="{{.ID}}"); if [[ ! -z "$running_container_ids" ]]; then docker kill $running_container_ids; fi'