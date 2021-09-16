# Imports
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
sys.path.append("data_processing")
import transforms
import sampling
import formatting

# %%
# =============================================================================
# =============================================================================
# =============================================================================

# Parameters

# =============================================================================
# Dataset generation
# =============================================================================
DATA_DIR = "data/arrays/"
DATASET_DIR = "data/datasets/"

# Interpolate to add frames and/or add pixels
interpolation = True
interp_size = (20, 100, 100)
interp_str = ""
if interpolation:
    interp_str = "_interp-{}-{}-{}".format(*interp_size)

# Other transforms
threshold = True
threshold_str = ""
if threshold:
    threshold_str = "_threshold"
flips = True
rotates = True
gauss_blur = True
gauss_noise = False
poisson_noise = True

# Split images into subimages?
subimages = False
sub_step = 10
sub_dim = 32
sub_str = ""
if subimages:
    sub_str = "_subims-step-{}-dim-{}".format(sub_step, sub_dim)

# Train, valid, test split
train_split = .8
valid_split = .2
test_split = 0
# Test runs
test_runs = [215, 214, 222, 609, 610, 447, 52, 43, 42, 80, 79, 78, 2, 3, 4, 5, 6]

# Save string
DESC_STR = "nufeb" + interp_str + threshold_str + sub_str
# =============================================================================
# =============================================================================
# =============================================================================

# Load the data
ims = []
test_ims = []
print("Loading data from: ", DATA_DIR)
print("-------------------------")
for file in glob(DATA_DIR + "run*.npy"):
    run_num = int(file.split(".")[-2].split("n")[-1])
    if run_num in test_runs:
        # Can't append them here since they are not in order
        continue
    else:
        print(file)
        ims.append(np.load(file))
for run_num in test_runs:
    test_ims.append(np.load(DATA_DIR + "run{}.npy".format(run_num)))
ims = np.stack(ims)
test_ims = np.stack(test_ims)
# Data shape is now: (num_runs, num_frames, pixel height, pixel width, num channels)

#-----------------------------------------------------------------------------
# Transforming the data
#-----------------------------------------------------------------------------
if interpolation:
    print("Interpolating images to make stacks of size: ", interp_size)
    new_ims = np.zeros((ims.shape[0], *interp_size, *ims.shape[len(interp_size)+1:]))
    for i in range(ims.shape[0]):
        new_ims[i] = transforms.interpolate_stack(ims[i], interp_size)
    ims = new_ims
    new_ims = "" # Hack to free the memory of new_ims
    new_ims = np.zeros((test_ims.shape[0], *interp_size, *test_ims.shape[len(interp_size)+1:]))
    for i in range(test_ims.shape[0]):
        new_ims[i] = transforms.interpolate_stack(test_ims[i], interp_size)
    test_ims = new_ims
    new_ims = "" # Hack to free the memory of new_ims

if threshold:
    print("Thresholding images")
    new_ims = np.zeros((*ims.shape[:-1], 1))
    for i in range(ims.shape[0]):
    #     ims[i] = transforms.threshold_stack(ims[i])
        new_ims[i] = transforms.grayscale_stack(ims[i])
    ims = new_ims
    new_ims = np.zeros((*test_ims.shape[:-1], 1))
    for i in range(test_ims.shape[0]):
    #     ims[i] = transforms.threshold_stack(ims[i])
        new_ims[i] = transforms.grayscale_stack(test_ims[i])
    test_ims = new_ims

# Flips, blurs, rotates, etc.
final_ims = [np.concatenate(ims)]
if flips:
    print("Flipping images")
    for i in range(ims.shape[0]):
        for flip in ["x", "y"]:
            final_ims.append(transforms.flip_stack(ims[i], direction=flip))
if rotates:
    print("Rotating images")
    for i in range(ims.shape[0]):
        for theta in [90, 180, 270]:
            final_ims.append(transforms.rotate_stack(ims[i], theta=theta))
if gauss_blur:
    print("Gaussian blurring images")
    for i in range(ims.shape[0]):
        final_ims.append(transforms.gaussian_blur_stack(ims[i], 3))
if gauss_noise:
    print("Gaussian noising images")
    for i in range(ims.shape[0]):
        final_ims.append(transforms.gaussian_noise_stack(ims[i]))
if poisson_noise:
    print("Poisson noising images")
    for i in range(ims.shape[0]):
        final_ims.append(transforms.poisson_noise_stack(ims[i]))

batch_size = ims.shape[1]
ims = np.concatenate(final_ims)
test_ims = np.concatenate(test_ims)

# Subimage split
if subimages:
    print("-------------------------")
    print("Splitting images into subimages of size {}x{} with steps of size {} between their centers".format(sub_dim, sub_dim, sub_step))
    ims = sampling.get_subimages(ims, sub_step, sub_dim, batch_size)
else:
    ims = sampling.get_batches(ims, batch_size)
    test_ims = sampling.get_batches(test_ims, batch_size)
# Data shape is now: (num_batches, num_frames, pixel height, pixel width, num channels)

# Shuffle dataset
rng = np.random.default_rng(seed=1)
rng.shuffle(ims)
#-----------------------------------------------------------------------------
# Generating a predRNN dataset
train_ims, valid_ims, _, train_clips, valid_clips, _, dims = \
    formatting.predrnn_format(ims, train_split, valid_split, test_split, batch_size)
# Use selected runs as test data
test_ims, _, _, test_clips, _, _, _ = \
    formatting.predrnn_format(test_ims, 1, 0, 0, batch_size)

# Saving the results
print("-------------------------")
print("Files saved: ")
train_str = DATASET_DIR + DESC_STR + "-train.npz"
valid_str = DATASET_DIR + DESC_STR + "-valid.npz"
test_str = DATASET_DIR + DESC_STR + "-test.npz"
print(train_str)
print(valid_str)
print(test_str)

np.savez(train_str, clips=train_clips, dims=dims, input_raw_data=train_ims)
np.savez(valid_str, clips=valid_clips, dims=dims, input_raw_data=valid_ims)
np.savez(test_str, clips=test_clips, dims=dims, input_raw_data=test_ims)

