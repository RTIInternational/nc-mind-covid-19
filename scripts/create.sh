#!/usr/bin/env bash

set -eu

source ./scripts/env.sh

python model_runs/src/create_runs.py
