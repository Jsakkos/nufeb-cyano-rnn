import numpy as np
import h5py
from glob import glob
import sys
sys.path.append("src/")
import area2d as a2d

# -----------------------------------------------------------------------------
# Testing numpy type error message
# -----------------------------------------------------------------------------
num_frames = 20
step = 10

for run_file in glob("../../data/nufeb/files/*.h5"):
    # Load run info
    scale = 1e6
    height = 100
    width = 100
    viz_scale = 20

    trajectory = h5py.File(run_file, "r")
    timesteps = a2d.get_timesteps(trajectory)
    ancestors = a2d.assign_ancestry(trajectory)

    ims = []
    for fnum in range(0, num_frames*step, step):
        ims.append(a2d.plot_colonies_at_time(timesteps[0][fnum], trajectory, scale, ancestors, height, width, viz_scale))
    ims = np.array(ims)
