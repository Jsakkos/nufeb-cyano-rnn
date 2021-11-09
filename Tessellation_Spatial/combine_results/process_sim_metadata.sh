#!/usr/bin/env bash

set -euo pipefail

find $1 -type d -name 'Run_*' -exec python get_run_metadata.py {} \;

