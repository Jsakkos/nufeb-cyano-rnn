import sys
sys.path.append("results_processing/")
from nufeb_populations import nufeb_populations
from gt_vs_pd_panels import gt_vs_pd_panels
from gt_vs_pd_video import gt_vs_pd_videos
from predrnn_accuracy import predrnn_accuracy

# --------------------------------------------------------------------------
# Parameters
RES_DIR = "results/validation_images/"
LOG_DIR = "results/logs/"
PLOTS_DIR = "results/images/"
VIDEOS_DIR = "results/videos/"
POP_DIR = "results/plots/"
ACC_DIR = "results/plots/"
BASE = "nufeb"

show_plots = True
save_plots = False
show_videos = True
save_videos = False
num_frames = 20
jump_size = 4

layers = 64
lr = 0.0003
threshold = True
interpolation = True
interp_sizes = [20, 100, 100]
subimages = False
sub_step = 10
sub_dim = 32
test = True

layer_str = "_layers-{}_lr-{}".format(layers, lr)
threshold_str = ""
if threshold:
    threshold_str = "_threshold"
interp_str = ""
if interpolation:
    interp_str = "_interp-{}-{}-{}".format(*interp_sizes)
sub_str = ""
if subimages:
    sub_str = "_subims-step-{}-dim-{}".format(sub_step, sub_dim)
test_str = ""
if test:
    test_str = "-test"

LOC = BASE + interp_str + sub_str + threshold_str + layer_str + test_str
print(LOC)
# --------------------------------------------------------------------------

# Accuracy plots from log
try:
    if test == False:
        predrnn_accuracy(LOG_DIR, ACC_DIR, LOC, show_plots, save_plots, layer_str)
except:
    pass
# Make panels
gt_vs_pd_panels(RES_DIR, PLOTS_DIR, LOC, num_frames, show_plots, save_plots, jump_size)
# Make videos
gt_vs_pd_videos(RES_DIR, VIDEOS_DIR, LOC, num_frames, show_videos, save_videos, jump_size)
# Make population curves
nufeb_populations(RES_DIR, POP_DIR, LOC, num_frames, show_plots, save_plots)
