#!/usr/bin/env bash

set -euo pipefail

experiment_dir=$1 

#"/home/joe/professional/research/NUFEB-cyanobacteria/data/exploratory/fourth_test_with_dist/"

find $experiment_dir -type d -wholename '*Run_[0-9]*_[0-9]*/Results/shape*' -exec echo {} > sim_dirs.txt \;

# choosing to do these as two separate blocks, one for atom size metadata
# and one for point pattern metadta.  Could interleave but messy. There
# is some repeated code which could probably be pulled into a function

# gather the atom size metadata
first_dir="$(head -n 1 sim_dirs.txt)"
header="$(head -n 1 $first_dir"/atom_sizes.csv")"
echo $header > gathered_atom_sizes.csv

for p in `cat sim_dirs.txt`
do
    p1="$(dirname "$p")"
    p2="$(dirname "$p1")"
    runname="$(basename "$p2")"
    file=$p"/atom_sizes.csv" 
    tail -n +2 $file  >> gathered_atom_sizes.csv
done

# gather the point pattern metadata
first_dir="$(head -n 1 sim_dirs.txt)"
header="$(head -n 1 $first_dir"/spatial_distribution.csv")"
echo $header > gathered_spatial_distributions.csv

for p in `cat sim_dirs.txt`
do
    p1="$(dirname "$p")"
    p2="$(dirname "$p1")"
    runname="$(basename "$p2")"
    file=$p"/spatial_distribution.csv" 
    tail -n +2 $file  >> gathered_spatial_distributions.csv
done
