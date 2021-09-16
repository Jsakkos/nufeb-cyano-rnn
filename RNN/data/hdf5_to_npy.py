import numpy as np
import h5py
import matplotlib.pyplot as plt
from glob import glob
import sys
sys.path.append("../nufebtools/src/")
import area2d as a2d

# -----------------------------------------------------------------------------
# This file is to convert NUFEB simulation hdf5 runs from 
# to numpy image arrays showing all the cells colored by ancestry
# -----------------------------------------------------------------------------

DATA_DIR = "files/"
SAVE_DIR = "arrays/"
num_frames = 20
step = 10

# Load run info
# From joe's voronoi_analysis demo
scale = 3e6 # px_pm - pixels per meter
sim_xspan = 1e-4
sim_yspan = 1e-4
height = int(sim_yspan*scale)
width = int(sim_xspan*scale)
viz_scale = 5 # Currently unused

for run_file in glob(DATA_DIR + "*.h5"):

    run_num = run_file[len(DATA_DIR):].split(".")[0][3:]
    print("File: ", run_file)

    trajectory = h5py.File(run_file, "r")
    timesteps = a2d.get_timesteps(trajectory)
    ancestors = a2d.assign_ancestry(trajectory)

    print("Number of time steps: ", len(timesteps[0]))
    ims = []
    if len(timesteps[0]) < num_frames*step:
        print("NOT ENOUGH STEPS: {} available, {} needed".format(len(timesteps[0]), num_frames*step))
        continue
    for fnum in range(0, num_frames*step, step):
        im = a2d.plot_colonies_at_time(timesteps[0][fnum], trajectory, scale, ancestors, height, width, viz_scale)
        ims.append(im)
        # im[im < 0] = 0
        # plt.imshow(im)
        # plt.show()
    ims = np.array(ims)

    np.save(SAVE_DIR + "run{}.npy".format(run_num), ims)
