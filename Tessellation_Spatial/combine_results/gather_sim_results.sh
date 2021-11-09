#!/usr/bin/env bash

set -euo pipefail

experiment_dir=$1 

#"/home/joe/professional/research/NUFEB-cyanobacteria/data/exploratory/fourth_test_with_dist/"

find $experiment_dir -type d -wholename '*Run_[0-9]*_[0-9]*/Results/shape*' -exec echo {} > sim_dirs.txt \;

first_dir="$(head -n 1 sim_dirs.txt)"
header="$(head -n 1 $first_dir"/run_areas_2d.csv")"
echo "RunID,"$header > gathered_sim_results.csv

for p in `cat sim_dirs.txt`
do
    p1="$(dirname "$p")"
    p2="$(dirname "$p1")"
    runname="$(basename "$p2")"
    file=$p"/run_areas_2d.csv" 
    tail -n +2 $file | sed -r "~s/^/$runname,/" - >> gathered_sim_results.csv
done
