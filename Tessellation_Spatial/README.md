# Analysis code for tessellation and spatial distribution 

There are three major subsections to these analyses

1. Generating the spatial point distributions and running the simulations
2. Postprocessing of the simulation results
3. Plotting and statistical analysis of the postprocessed results

Note that steps 1 and 2 are rather resource intensive (especially step 1).  We provide the code here for reproducibility and examination, but expect most interested parties will want to execute code starting at Step 3 using the included results of the postprocessing step, located in the ```data``` subdirectory of ```analysis```.

## Generating and running the simulations

The ```spatial``` directory contains the R script and project used to generate a CSV file defining the locations of initial seed sites for all spatial pattern distributions. 

The CSV file can be used to create the atom.in files for each NUFEB run. The runs used in the paper were submitted to an HPC cluster using the SLURM workflow management tool.

## Postprocessing simulation results

After running the simulations, a suite of python and shell scripts can be used to aggregate relevant simulation results into a single CSV file which contains the final colony area resulting from each initial seed location as well as metadata concerning the indivual run (spatial pattern, number of initial seeds, etc).

The suite of scripts is located in the ```combine_results``` directory and an example of the order in which they are called is located in ```call_pipeline_example.sh```. 

The scripts rely upon a prototype nufebtools package, located in the directory of the same name.  This package represents the state of the code at the time of the analysis, with the active git repository at https://github.com/joeweaver/nufebtools. Note that this is distinct from the NUFEB-tools code at https://github.com/Jsakkos/nufeb_tools, although plans are underway to merge the two codebases.

Note that the Voronoi tesselation figures were also produced using the nufebtools package by running against selected HDF5-formatted outputs from some simulations.

Additional data files describing the centroid locations of each voronoi facet and their distances to the associated seed locations were calculated as in extra_posAdditional data files describing the centroid locations of each voronoi facet and their distances to the associated seed locations were calculated using the scripts in  ```extra_post```.

## Analysis of the processed data

Analysis and figure generation is performed using the R scripts within the ```analysis``` directory. The scripts may be run in any order and depend on the csv files in the ```data``` subdirectory, which are the results of the simulation and processing in steps 1 and 2.

The resulting figures are saved in the ```output``` subdirectory while the results of various regression fits were recorded interactively.
