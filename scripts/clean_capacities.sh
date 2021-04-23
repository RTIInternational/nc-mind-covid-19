#!/bin/bash
set -eu
source ./scripts/env.sh

parallel --sshloginfile scripts/nodes.txt --onall ::: "cd $RUN_PATH && docker-compose run --rm covid rm -rf ./data/locations/"
parallel --sshloginfile scripts/nodes.txt --onall ::: "cd $RUN_PATH && docker-compose run --rm covid rm -rf ./data/transitions/"
