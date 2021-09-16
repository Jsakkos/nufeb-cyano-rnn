import numpy as np
import torch
import sys
sys.path.append("..")
from core.models.model_factory import Model
from argparse import Namespace

# %%
# Define arguments
args = {
    'is_training': 1,
    'device': 'cpu:0',
    'dataset_name': 'mnist',
    'train_data_paths': '/home/connor/GDrive/SCGSR/data/datasets/wells-1_microns-30_grayscale_red-train.npz',
    'valid_data_paths': '/home/connor/GDrive/SCGSR/data/datasets/wells-1_microns-30_grayscale_red-valid.npz',
    'save_dir': 'checkpoints/bacteria_predrnn',
    'gen_frm_dir': 'results/bacteria_predrnn',
    'input_length': 7,
    'total_length': 14,
    'img_width': 24,
    'img_channel': 1,
    'model_name': 'predrnn',
    'pretrained_model': '',
    'num_hidden': '64,64,64,64',
    'filter_size': 5,
    'stride': 1,
    'patch_size': 4,
    'layer_norm': 1,
    'decouple_beta': 0.1,
    'reverse_scheduled_sampling': 0,
    'r_sampling_step_1': 25000,
    'r_sampling_step_2': 50000,
    'r_exp_alpha': 5000,
    'scheduled_sampling': 1,
    'sampling_stop_iter': 50000,
    'sampling_start_value': 1.0,
    'sampling_changing_rate': 2e-05,
    'lr': 0.001,
    'reverse_input': 1,
    'batch_size': 2,
    'max_iterations': 80000,
    'display_interval': 100,
    'test_interval': 5000,
    'snapshot_interval': 5000,
    'num_save_samples': 10,
    'n_gpu': 1,
    'visual': 0,
    'visual_path': './decoupling_visual'
}

# So that args dictionary elements can be accessed as objects
# e.g. args["n_gpu"] -> args.n_gpu
args = Namespace(**args)

# Create model
model = Model(args)

# Load data
train_data = np.load(args.train_data_paths)
valid_data = np.load(args.valid_data_paths)

# Try forward pass
ims = torch.from_numpy(train_data['input_raw_data'][:args.input_length])
ims = ims.permute(0, 2, 3, 1)
