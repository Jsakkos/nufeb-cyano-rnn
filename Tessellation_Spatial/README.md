# Analysis code for tessellation and spatial distribution 

There are three major subsections to these analyes

1. Generating the spatial point distributions and running the simulations
2. Postprocessing of the simulation results
3. Plotting and statistical analysis of the postprocessed results

## Generating and running the simulations

The ```spatial``` directory contains the R script and project used to generate a CSV file defining the locations of initial seed sites for all spatial pattern distributions. 

The CSV file can be used to create the atom.in files for each NUFEB run. The runs used in the paper were submitted to an HPC cluster using the SLURM workflow management tool.


