import numpy as np

def predrnn_format(ims, train_split, valid_split, test_split, batch_size):
    # Put in predrnn format
    train_size = batch_size // 2
    test_size = batch_size // 2

    train_cut = int(train_split*ims.shape[0])
    valid_cut = train_cut + int(valid_split*ims.shape[0])
    test_cut = valid_cut + int(test_split*ims.shape[0])
    if train_cut != 0:
        train_ims = np.concatenate(ims[0:train_cut])
        train_ims = np.moveaxis(train_ims, -1, 1)
    else:
        train_ims = np.zeros(1)
    if valid_cut != train_cut:
        valid_ims = np.concatenate(ims[train_cut:valid_cut])
        valid_ims = np.moveaxis(valid_ims, -1, 1)
    else:
        valid_ims = np.zeros(1)
    if test_cut != valid_cut:
        test_ims = np.concatenate(ims[valid_cut:])
        test_ims = np.moveaxis(test_ims, -1, 1)
    else:
        test_ims = np.zeros(1)

    # Clips
    train_inds = np.arange(0, ims.shape[0]*batch_size, train_size+test_size)
    test_inds = np.arange(train_size, ims.shape[0]*batch_size, train_size+test_size)
    train_steps = train_size * np.ones_like(train_inds)
    test_steps = test_size * np.ones_like(test_inds)
    clips = np.stack([np.vstack([train_inds, train_steps]).T, np.vstack([test_inds, test_steps]).T], axis=0)

    train_clips = clips[:, 0:train_cut, :]
    valid_clips = clips[:, train_cut:valid_cut, :]
    valid_clips[:, :, 0] -= train_cut*batch_size
    test_clips = clips[:, valid_cut:, :]
    test_clips[:, :, 0] -= valid_cut*batch_size

    # Dims
    dims = np.array([[*train_ims.shape[1:]]])

    return train_ims, valid_ims, test_ims, train_clips, valid_clips, test_clips, dims
