#!/usr/bin/env bash

set -euo pipefail

./process_sim_results.sh $1 $2
./process_sim_metadata.sh $1
./gather_sim_results.sh $1
./gather_sim_metadata.sh $1
python combine_gathered_results_metadata.py
