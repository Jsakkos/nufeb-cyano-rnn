import os
import h5py
from nufebtools import area2d
# This cell is tagged `parameters` for use with papermill
# controlling variables, should eventually be arguments
rundir = '/home/joe/professional/research/NUFEB-cyanobacteria/data/exploratory/fourth_test_with_dist/distributions/Run_16_1'
resultsdir = 'results'
outdir = 'shape_metrics'
genfig_colony_growth = False
write_final_data = False
write_all_time_metrics_data = False

# flag to keep us from doing stuff we only want done when papermill is
# calling
# the notebook, such as saving files
# implication is that papermill should definitely pass this parameter as
# True
using_papermill = False


# hdf5 encoded file with all dumped NUFEB output
h5file = 'trajectory.h5'

# specifying pixels per meter instead of specifying image width and height
# this allows us to analyze virtual images at the same resolution as the
# real lab images
px_pm = 3e6

# can be externally specified, ideally by something which understands the
# input atom file
# x and y dimensions in meters of simulation
sim_xspan = 1e-4
sim_yspan = 1e-4

im_height = int(sim_yspan*px_pm)
im_width = int(sim_xspan*px_pm)

infile = os.path.join(rundir, h5file)


# print diagnostic output
DEBUG_PRINT = False
traj = h5py.File(infile, 'r')

# %%
ancestry = area2d.assign_ancestry(traj)

# %%
colony_morphologies = area2d.get_colony_morphologies_at_times(
    area2d.get_timesteps(traj)[0], ancestry, traj, px_pm, im_height, im_width)
facet_morphologies = area2d.get_facet_morphologies(traj,px_pm,im_height,im_width)

#%%
import cv2
facet_img = area2d.graph_facets(traj, px_pm, im_height, im_width, 5)
#cv2.imwrite("test_facet.png",facet_img)
#%%
combined_morphologies = area2d.combine_morphologies(colonies=colony_morphologies,facets=facet_morphologies)

area2d.plot_colony_growth(colony_morphologies)

# %%
import matplotlib.pyplot as plt
raw_img = area2d.plot_colonies_at_time(34900, traj, px_pm, ancestry, im_height, im_width, 5)
plt.imshow(cv2.flip(raw_img, 1))
