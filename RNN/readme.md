# Using predRNN as a metamodel

In order to accelerate simulation results, we tested the use of a recurrent neural network as a "metamodel" which can approximate the true agent based simulation for prediction of future states. This work makes use of the pytorch implementation of PredRNN at https://github.com/thuml/predrnn-pytorch as well as the nufebtools at https://github.com/joeweaver/nufebtools.

## Directory overview
- `data`: Contains simulation data used to create simulation images and a file `hdf5_to_npy.py` to convert from NUFEB hdf5 output to images using nufebtools
- `data_processing`: Contains tools for creating the training and testing datasets for predRNN
- `exploration`: Contains tools to visualize colonies in the simulation images
- `nufebtools`: Submodule of nufebtools repository
- `predrnn-pytorch`: Submodule of predRNN pytorch repository
- `results`: For storing results from predRNN and subsequent analysis
- `results_processing`: Contains tools for processing results and generating figures 

## Requirements
A `requirements.txt` and `environment.yml` files are included for pip and conda dependencies respectively. However, the main required packages are:
- numpy
- matplotlib
- pytorch (compiled with cuda if you are using a GPU)
- scikit-image
- atomai

**Note:** The file `predrnn-pytorch/core/trainer.py` has been modified to use `skimage.metrics.structural_similarity()` in place of `skimage.measure.compare_ssim()` in order to use more recent versions of scikit-image.

## Instructions
1. Adjust parameters in `generate_datset.py` then run to create a dataset in the chosen directory
2. Adjust the parameters in `train_predrnn.sh` then run to train and validate a predRNN model with the training data from step 1
3. Adjust the parameters in `test_predrnn.sh` then run to generate predictions using the test dataset created in step 1
4. Adjust the parameters in `analyze_results.py` then run to generate plots
5. Adjust the parameters in `paper_plots.py` to generate plots seen in the paper

