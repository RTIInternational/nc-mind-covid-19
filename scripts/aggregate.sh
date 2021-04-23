#!/usr/bin/env bash

set -eu

source ./scripts/env.sh

python model_runs/src/aggregate_results.py
