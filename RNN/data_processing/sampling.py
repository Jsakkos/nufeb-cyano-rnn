import numpy as np
from atomai import utils

def get_subimages(ims, sub_step, sub_dim, batch_size):
    """
    Extract subimages from stack of images

    Args:
    =====
    - ims: a stack of images either 3d (frames, height, width) or 4d
        (frames, height, width, channels)
    - sub_step: the distance between the center of each subimage. Smaller means
        more overlap between subimages
    - sub_dim: the height/width of the subimages
    - batch_size: the batch_size of the images

    Output:
    =======
    - new_ims: ims as shape (num_batches, batch_size, height, width, channels)
    """
    if len(ims.shape) == 3:
        coords = utils.get_coord_grid(ims, sub_step)
    else:
        coords = utils.get_coord_grid(ims[..., 0], sub_step)
    temp_tiles, tile_coords, frame_nums = utils.extract_subimages(ims, coords, sub_dim)
    num_subimages = np.sum(frame_nums == 0)
    if len(ims.shape) == 3:
        new_ims = np.zeros((num_subimages, ims.shape[0], sub_dim, sub_dim))
    else:
        new_ims = np.zeros((num_subimages, ims.shape[0], sub_dim, sub_dim, 3))
    for i in range(num_subimages):
        new_ims[i, ...] = temp_tiles[i:temp_tiles.shape[0]:num_subimages]

    # Split up into batches
    new_ims = np.concatenate([new_ims[:, i:i+batch_size, ...] for i in range(0, new_ims.shape[1], batch_size)])
    return new_ims

def get_batches(ims, batch_size):
    """
    Extract batches from stack of images

    Args:
    =====
    - ims: a stack of images either 3d (frames, height, width) or 4d
        (frames, height, width, channels)
    - batch_size: the batch_size of the images

    Output:
    =======
    - new_ims: ims as shape (num_batches, batch_size, height, width, channels)
    """
    # Split up into batches
    new_ims = np.array([ims[i:i+batch_size, ...] for i in range(0, ims.shape[0], batch_size)])
    return new_ims

