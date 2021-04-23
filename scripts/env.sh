#!/bin/bash

# Repository code and run parameters will be copied to this
# directory on every machine used for running models (potentially including
# the local machine)
export RUN_PATH="/tmp/$(whoami)/covid19"
