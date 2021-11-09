#!/usr/bin/env bash

set -euo pipefail

find $1 -type d -name 'Run_*' -exec papermill -p interactive False -p dry_run False -p save_plot_colony_growth True -p rundir {} $2 /dev/null \;

