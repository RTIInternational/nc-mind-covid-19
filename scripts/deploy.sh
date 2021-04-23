#!/bin/bash

set -eu

source ./scripts/env.sh

parallel --sshloginfile scripts/nodes.txt --onall ::: "mkdir -p $RUN_PATH"
for server in $(cat ./scripts/nodes.txt)
do
    if [[ "$server" == ":" ]]; then
        # Special GNU parallel value representing localhost
        dest_loc=$RUN_PATH
    else
        dest_loc="$server:$RUN_PATH"
    fi
    rsync -az --keep-dirlinks --no-p --filter '+ data/synthetic_population/*' --filter '+ data/IDs/*' --filter '+ data/transitions/*' --filter=':- .gitignore' --filter '- .git/' --progress ./ $dest_loc
done

parallel --sshloginfile scripts/nodes.txt --onall ::: "hostname; cd $RUN_PATH; docker-compose build covid"
