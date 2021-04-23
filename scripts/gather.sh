#!/bin/bash

set -eu

source ./scripts/env.sh

for server in $(cat ./scripts/nodes.txt)
do
    if [[ "$server" == ":" ]]; then
       # Special GNU parallel value representing localhost
       src_loc=$RUN_PATH
    else
        src_loc="$server:$RUN_PATH"
    fi
    rsync -r ${src_loc}/model_runs/parallel_runs/ model_runs/parallel_runs/
done
