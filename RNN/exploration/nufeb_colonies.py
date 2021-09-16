import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import measure
from skimage import filters
from skimage import color
import skimage

# %%
# Can we identify the nufeb clusters?

# %%
# Load data
ts = 19
im = np.load("../../data/nufeb/arrays/run87.npy")
col_colors = np.delete(np.unique(np.round(color.rgb2gray(im[0]), 5)), 0)
num_colonies = len(col_colors)
red = im[ts, :, :, 0]
green = im[ts, :, :, 1]
green[green < 0] = 0
plt.figure(figsize=(12, 5))
plt.subplot(131)
plt.imshow(red)
plt.colorbar()
plt.title("Red raw data")
plt.subplot(132)
plt.imshow(green)
plt.colorbar()
plt.title("Green raw data")
plt.subplot(133)
plt.imshow(im[ts])
plt.colorbar()
plt.title("Full data")
plt.show()

# Convert image to grayscale (hopefully preserving uniqueness of colony colors)
im = color.rgb2gray(im)
plt.imshow(im[ts], cmap="gray")
plt.title("Full data")
plt.show()

# %%
# SKIMAGE utilities
filters.try_all_threshold(im[ts])
plt.suptitle("Im threshold results - t: {}".format(ts))
plt.show()

# %%
print("Num colonies: ", num_colonies)

# %%
# How much of each species is present?
tot_mass = 0
for ci in range(len(col_colors)):
    num_col = np.sum(np.abs(im - col_colors[ci]) < 1e-5)
    print("Colony ", ci, ": ", num_col)
    tot_mass += num_col
print("Total mass: ", tot_mass)
print("Total mass direct: ", np.sum(im > 1e-2))

# %%
# Median-normed colony area
col_meds = []
for t in range(im.shape[0]):
    areas = []
    for ci in range(len(col_colors)):
        num_col = np.sum(np.abs(im[t] - col_colors[ci]) < 1e-5)
        areas.append(num_col)
    area_med = np.median(areas)
    col_meds.append((np.array(areas) - area_med) / area_med)
col_meds = np.array(col_meds)
print(np.sum(col_meds, axis=0))

# %%
# What shape are the colonies?
thresh = filters.threshold_otsu(im)
contours = measure.find_contours(im[ts], thresh)
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(im[ts], cmap="gray")
plt.colorbar()
plt.title("Green -- thresh: {}".format(thresh))
for c in contours:
    plt.plot(c[:, 1], c[:, 0], color="red")
plt.subplot(122)
plt.imshow(im[ts], cmap="gray")
plt.colorbar()
plt.title("Full data")
plt.show()

