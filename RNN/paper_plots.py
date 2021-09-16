import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lpips
import torch
from glob import glob
from scipy.stats import ks_2samp
from skimage import measure
from itertools import product
import sys
sys.path.append("results_processing/")
from predrnn_accuracy import predrnn_accuracy

# =============================================================================
# Plots for the nufeb ABM paper:
# - Histogram of the median normed colony areas for the test runs in each case
#       compared against histograms from simulation
# - Total population of species in simulation vs predicted
# =============================================================================

# %%
# =========================
# Load data
DATA_DIR = "results/validation_images/"
LOG_DIR = "results/logs/"
LOC = "nufeb_interp-20-100-100_threshold_layers-64_lr-0.0003-test"
TRAIN_LOC = "nufeb_interp-20-100-100_threshold_layers-64_lr-0.0003"
num_frames = 20
# =========================

batches = []
for batch_dir in glob(DATA_DIR + LOC + "/*"):
    if "test" in batch_dir:
        batches.append(batch_dir.split("/")[-1])
    else:
        batches.append(int(batch_dir.split("/")[-1]))
batches.sort()
tests = []
for test_dir in glob(DATA_DIR + LOC + "/{}/*".format(batches[0])):
    tests.append(int(test_dir.split("/")[-1]))
tests.sort()

dir_num = batches[0]
# =========================

# %%
loss_fn_alex = lpips.LPIPS(net='alex')
# Median normed colony area
test_labels = 6*["aggregate"] + 6*["regular"] + 5*["random"]
agg_areas = []; reg_areas = []; rand_areas = []
p_agg_areas = []; p_reg_areas = []; p_rand_areas = []
agg_sizes = []; reg_sizes = []; rand_sizes = []
p_agg_sizes = []; p_reg_sizes = []; p_rand_sizes = []
agg_mse = []; reg_mse = []; rand_mse = []
agg_lpips = []; reg_lpips = []; rand_lpips = []
for test_num, test_lab in zip(tests, test_labels):
    batch_dir = DATA_DIR + LOC + "/{}/{}".format(dir_num, test_num)
    img_array = []
    p_img_array = []
    gt_img = []
    pd_img = []

    for file_num in range(1,num_frames):
        gt_img.append(batch_dir + "/gt" + str(file_num) + ".png")

    for file_num in range(num_frames//2 + 1, 3*num_frames//2):
        pd_img.append(batch_dir + "/pd" + str(file_num) + ".png")

    for gt_file, pd_file in zip(gt_img, pd_img):
        if len(plt.imread(gt_file).shape) == 2:
            img_array.append(plt.imread(gt_file))
            p_img_array.append(plt.imread(pd_file))
        else:
            img_array.append(plt.imread(gt_file)[:, :, ::-1])
            p_img_array.append(plt.imread(pd_file)[:, :, ::-1])
    # To only consider predicted frames
    img_array = np.array(img_array)[num_frames//2:]
    p_img_array = np.array(p_img_array)[num_frames//2:]
    # img_array = np.array(img_array)
    # p_img_array = np.array(p_img_array)

    if len(img_array.shape) == 4:
        img_array = np.linalg.norm(img_array, axis=-1)
        p_img_array = np.linalg.norm(p_img_array, axis=-1)
    tot_area = np.prod(img_array.shape[1:])

    torch_ims = torch.from_numpy(img_array[:, np.newaxis, :, :]).float()
    torch_p_ims = torch.from_numpy(p_img_array[:, np.newaxis, :, :]).float()
    if test_lab == "aggregate":
        agg_sizes.append(np.sum(img_array, axis=(1, 2))/tot_area)
        p_agg_sizes.append(np.sum(p_img_array, axis=(1, 2))/tot_area)
        agg_mse.append(np.sum((img_array - p_img_array)**2, axis=(1,2))/img_array.shape[0])
        agg_lpips.append(loss_fn_alex(torch_ims, torch_p_ims).flatten().tolist())
    elif test_lab == "regular":
        reg_sizes.append(np.sum(img_array, axis=(1, 2))/tot_area)
        p_reg_sizes.append(np.sum(p_img_array, axis=(1, 2))/tot_area)
        reg_mse.append(np.sum((img_array - p_img_array)**2, axis=(1,2))/img_array.shape[0])
        reg_lpips.append(loss_fn_alex(torch_ims, torch_p_ims).flatten().tolist())
    elif test_lab == "random":
        rand_sizes.append(np.sum(img_array, axis=(1, 2))/tot_area)
        p_rand_sizes.append(np.sum(p_img_array, axis=(1, 2))/tot_area)
        rand_mse.append(np.sum((img_array - p_img_array)**2, axis=(1,2))/img_array.shape[0])
        rand_lpips.append(loss_fn_alex(torch_ims, torch_p_ims).flatten().tolist())

    col_seeds = np.vstack(np.where(img_array[0] != 0)).T
    seed_arr = np.zeros_like(img_array[0])
    for seed in col_seeds:
        seed_arr[seed[0], seed[1]] = 1
    seed_labels = measure.label(seed_arr)
    num_seeds = len(np.unique(seed_labels))-1
    col_seeds = np.zeros((num_seeds, 2))
    for i in range(1, num_seeds+1):
        temp_seeds = np.vstack(np.where(seed_labels == i))
        col_seeds[i-1] = np.mean(temp_seeds, axis=1).astype(int)

    labels = np.zeros_like(img_array)
    p_labels = np.zeros_like(img_array)
    for ti in range(img_array.shape[0]):
        all_cells = np.vstack(np.where(img_array[ti] != 0)).T
        p_all_cells = np.vstack(np.where(p_img_array[ti] != 0)).T
        for cell in all_cells:
            cell_dists = np.linalg.norm(cell - col_seeds, axis=1)
            col_label = np.argmin(cell_dists)
            closest_cell = col_seeds[col_label]
            labels[ti, cell[0], cell[1]] = col_label + 1
        for cell in p_all_cells:
            cell_dists = np.linalg.norm(cell - col_seeds, axis=1)
            col_label = np.argmin(cell_dists)
            closest_cell = col_seeds[col_label]
            p_labels[ti, cell[0], cell[1]] = col_label + 1

    # Count colonies
    cols_sizes = []
    p_cols_sizes = []
    for i in range(1, len(col_seeds)+1):
        col_sizes = []
        p_col_sizes = []
        for ti in range(img_array.shape[0]):
            col_sizes.append(np.sum(img_array[ti][labels[ti] == i]))
            p_col_sizes.append(np.sum(p_img_array[ti][p_labels[ti] == i]))
        cols_sizes.append(col_sizes)
        p_cols_sizes.append(p_col_sizes)

    cols_sizes = np.array(cols_sizes)
    p_cols_sizes = np.array(p_cols_sizes)

    col_meds = np.median(cols_sizes, axis=0)
    p_col_meds = np.median(p_cols_sizes, axis=0)
    if test_lab == "aggregate":
        agg_areas.append((cols_sizes - col_meds)/col_meds)
        p_agg_areas.append((p_cols_sizes - p_col_meds)/p_col_meds)
    elif test_lab == "regular":
        reg_areas.append((cols_sizes - col_meds)/col_meds)
        p_reg_areas.append((p_cols_sizes - p_col_meds)/p_col_meds)
    elif test_lab == "random":
        rand_areas.append((cols_sizes - col_meds)/col_meds)
        p_rand_areas.append((p_cols_sizes - p_col_meds)/p_col_meds)

agg_areas = np.vstack(agg_areas)
p_agg_areas = np.vstack(p_agg_areas)
reg_areas = np.vstack(reg_areas)
p_reg_areas = np.vstack(p_reg_areas)
rand_areas = np.vstack(rand_areas)
p_rand_areas = np.vstack(p_rand_areas)

agg_sizes = np.vstack(agg_sizes)
reg_sizes = np.vstack(reg_sizes)
rand_sizes = np.vstack(rand_sizes)

# %%
# Plot the final median normed colony areas
# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = cm.get_cmap("Set1").colors
bins = 30
time = 8
fig, ax = plt.subplots(3,1,sharex=True, figsize=(8, 6))
ax[0].hist(agg_areas[:, time], cumulative=False, density=True, bins=bins, alpha=.8, label="Simulation", color=colors[0])
ax[0].hist(p_agg_areas[:, time], cumulative=False, density=True, bins=bins, alpha=.8, label="Predicted", color=colors[1])
ax[0].set_title("Aggregate")
ax[0].legend(fontsize=14)

ax[1].hist(reg_areas[:, time], cumulative=False, density=True, bins=bins//3, alpha=.8, label="Simulation", color=colors[0])
ax[1].hist(p_reg_areas[:, time], cumulative=False, density=True, bins=bins//3, alpha=.8, label="Predicted", edgecolor=None, color=colors[1])
ax[1].set_title("Regular", )
# ax[1].legend()

ax[2].hist(rand_areas[:, time], cumulative=False, density=True, bins=bins//3, alpha=.8, label="Simulation", color=colors[0])
ax[2].hist(p_rand_areas[:, time], cumulative=False, density=True, bins=bins//3, alpha=.8, label="Predicted", color=colors[1])
ax[2].set_title("Random")
# ax[2].legend()
for a in ax:
    for label in (a.get_xticklabels() + a.get_yticklabels()):
        label.set_fontsize(14)

plt.xlabel("Median-normed colony areas (final prediction time)".format(time), fontsize=18)
plt.tight_layout()
# plt.savefig("results/plots/mnca_hist-{}.png".format(time), bbox_inches="tight")
plt.show()

# %%
# Plot total population growth
plt.figure(figsize=(8, 6))
colors = cm.get_cmap("Set1").colors
plt.plot(np.sum(agg_sizes, axis=0), color=colors[0], label="Aggregate", lw=3)
plt.plot(np.sum(p_agg_sizes, axis=0), color=colors[0], ls="--", label="Prediction", lw=3)
plt.plot(np.sum(reg_sizes, axis=0), color=colors[1], label="Regular", lw=3)
plt.plot(np.sum(p_reg_sizes, axis=0), color=colors[1], ls="--", label="Prediction", lw=3)
plt.plot(np.sum(rand_sizes, axis=0), color=colors[2], label="Random", lw=3)
plt.plot(np.sum(p_rand_sizes, axis=0), color=colors[2], ls="--", label="Prediction", lw=3)
plt.legend(fontsize=18)
plt.xlabel("Prediction Time", fontsize=24)
plt.ylabel("Total population area", fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.title("Global population comparison")
# plt.savefig("results/plots/population.png", bbox_inches="tight")
plt.show()

# %%
# Plot mse difference
plt.figure(figsize=(8, 6))
colors = cm.get_cmap("Set1").colors
plt.plot(np.sum(agg_mse, axis=0), color=colors[0], label="Aggregate", lw=3)
plt.plot(np.sum(reg_mse, axis=0), color=colors[1], label="Regular", lw=3)
plt.plot(np.sum(rand_mse, axis=0), color=colors[2], label="Random", lw=3)
plt.legend(fontsize=18)
plt.xlabel("Prediction Time", fontsize=24)
plt.ylabel("Mean Squared Error", fontsize=24)
# plt.title("Prediction error over time")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.savefig("results/plots/mse.png", bbox_inches="tight")
plt.show()

# %%
# Plot lpips difference
plt.figure(figsize=(8, 6))
colors = cm.get_cmap("Set1").colors
plt.plot(np.sum(agg_lpips, axis=0), color=colors[0], label="Aggregate", lw=3)
plt.plot(np.sum(reg_lpips, axis=0), color=colors[1], label="Regular", lw=3)
plt.plot(np.sum(rand_lpips, axis=0), color=colors[2], label="Random", lw=3)
plt.legend(fontsize=18)
plt.xlabel("Prediction Time", fontsize=24)
plt.ylabel("LPIPS (Perceptual Similarity)", fontsize=24)
# plt.title("Prediction error over time")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.savefig("results/plots/lpips.png", bbox_inches="tight")
plt.show()

# %%
# Scatter plot mnca of individual colonies in test -- show how it isnt always
# right, but mostly gets at least correlation (large mnca simulation -> large
# mnca prediction)
# m_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
m_colors = cm.get_cmap("Set1").colors
# markers = ["d", "v", "s", "*", "^", "d", "v", "s", "*", "^"]
markers = ["o"]
styles = list(product(markers, m_colors))
# np.random.shuffle(styles)

plt.figure(figsize=(8, 8))
num_points = 20
ref = [np.min(agg_areas[:num_points, -1]), np.max(p_agg_areas[:num_points, -1])]
plt.plot(ref, ref, ls="--", color="black", label="Exact")
plt.hlines(0, ref[0], ref[1], color="black", alpha=.6)
plt.vlines(0, ref[0], ref[1], color="black", alpha=.6)
for ci in range(num_points):
    i_size = agg_areas[ci, -1]
    p_size = p_agg_areas[ci, -1]
    plt.scatter([i_size], [p_size], marker=styles[ci%len(styles)][0], color=styles[ci%len(styles)][1], label="Colony {}".format(ci+1), zorder=100, edgecolor="black")
plt.fill_between([0, ref[1]], 0, ref[1], color="green", alpha=.2)
plt.fill_between([ref[0], 0], 0, ref[1], color="red", alpha=.2)
plt.fill_between([0, ref[1]], ref[0], 0, color="red", alpha=.2)
plt.fill_between([ref[0], 0], ref[0], 0, color="green", alpha=.2)
# plt.title("Comparison of final colony size")
plt.xlabel("Simulation median-normalized colony size", fontsize=14)
plt.ylabel("Predicted median-normalized colony size", fontsize=14)
plt.legend(fontsize=9)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.savefig("results/plots/mnca_scatter.png", bbox_inches="tight")
plt.show()

# Plot training mse and LPIPS
predrnn_accuracy(LOG_DIR, "results/plots/", TRAIN_LOC, True, False, "layers-64_lr-0.0003") 
