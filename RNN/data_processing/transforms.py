import numpy as np
from skimage.filters import threshold_otsu
from skimage.util import random_noise
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
import scipy.interpolate as interp
import cv2
from scipy.ndimage import gaussian_filter

def interpolate_stack(ims, sizes=(14, 64, 64)):
    """
    Use cv2 bilinear interpolation to resize images in a stack. Also use
    scipy.interpolate to augment the number of images in time.

    Args:
    =====
    ims - Stack of images: 3d - (frames, height, wid) or 4d - (frames, height, wid, channels)
    sizes - The size of the output interpolations, 3 tuple (num frames, height, wid)

    Outputs:
    ========
    new_ims - Stack of images of same number of dimensions as input but new sizes
    """
    if len(ims.shape) == 3:
        result = np.zeros((ims.shape[0], *sizes[1:]))
        new_ims = np.zeros(sizes)
    else:
        result = np.zeros((ims.shape[0], *sizes[1:], ims.shape[-1]))
        new_ims = np.zeros((*sizes, ims.shape[-1]))

    # Interpolate in space
    if np.any(np.array(ims.shape[2:]) - np.array(sizes[1:]) > 0): # Are we downsizing?
        inter_type = cv2.INTER_AREA
    else:
        inter_type = cv2.INTER_LINEAR
    for i in range(ims.shape[0]):
        result[i] = cv2.resize(ims[i], dsize=sizes[1:], interpolation=inter_type)

    # Interpolate in time
    steps = sizes[0]
    ts = np.arange(ims.shape[0])
    new_ts = np.linspace(0, len(ims)-1, steps)
    for i in range(result.shape[1]):
        for j in range(result.shape[2]):
            if len(ims.shape) == 3:
                temp_f = interp.interp1d(ts, result[:, i, j])
                new_ims[:, i, j] = temp_f(new_ts)
            else:
                for k in range(ims.shape[3]):
                    temp_f = interp.interp1d(ts, result[:, i, j, k])
                    new_ims[:, i, j, k] = temp_f(new_ts)
    return new_ims

def threshold_stack(ims):
    """
    Use Otsu thresholding from scikit image on stack of images: 3d -
        (frames, height, wid) or 4d - (frames, height, wid, channels)

    Args:
    =====
    ims - Stack of images: 3d - (frames, height, wid) or 4d - (frames, height, wid, channels)
    sizes - The size of the output interpolations, 3 tuple (num frames, height, wid)

    Outputs:
    ========
    new_ims - Stack of images of same number of dimensions as input but new sizes
    """
    new_ims = np.zeros_like(ims)
    thresh = threshold_otsu(ims.flatten())
    if len(ims.shape) == 3:
        # thresh = threshold_otsu(ims.flatten())
        new_ims[ims > thresh] = 1
    else:
        for k in range(ims.shape[-1]):
            # thresh = threshold_otsu(ims[..., k].flatten())
            new_ims[..., k][ims[..., k] > thresh] = 1
    return new_ims

def gaussian_noise_stack(ims, size=.01):
    """
    Apply noise to each channel of image stack dependent on var of stack

    Args:
    =====
    ims - Stack of images: 3d - (frames, height, wid) or 4d - (frames, height, wid, channels)
    size - the percentage of var of data to use as gaussian var

    Outputs:
    ========
    new_ims - Stack of images of same number of dimensions as input but new sizes
    """
    if len(ims.shape) == 3:
        noise_vars = [size * np.sqrt(np.std(ims))]
    else:
        noise_vars = [size * np.sqrt(np.std(ims[..., k])) for k in range(ims.shape[-1])]
    new_ims = np.zeros_like(ims)
    for i in range(ims.shape[0]):
        if len(ims.shape) == 3:
            new_ims[i] = random_noise(ims[i], mode="gaussian", var=noise_vars[0])
        else:
            for k in range(ims.shape[-1]):
                new_ims[i, ..., k] = random_noise(ims[i, ..., k], mode="gaussian", var=noise_vars[k])
    return new_ims

def poisson_noise_stack(ims):
    """
    Apply Poisson noise to each channel of image stack

    Args:
    =====
    ims - Stack of images: 3d - (frames, height, wid) or 4d - (frames, height, wid, channels)

    Outputs:
    ========
    new_ims - Stack of images of same number of dimensions as input but new sizes
    """
    new_ims = np.zeros_like(ims)
    for i in range(ims.shape[0]):
        if len(ims.shape) == 3:
            new_ims[i] = random_noise(ims[i], mode="poisson")
        else:
            for k in range(ims.shape[-1]):
                new_ims[i, ..., k] = random_noise(ims[i, ..., k], mode="poisson")
    return new_ims

def gaussian_blur_stack(ims, size=None):
    """
    Apply gaussian blurring to each channel of image stack dependent on var of stack

    Args:
    =====
    ims - Stack of images: 3d - (frames, height, wid) or 4d - (frames, height, wid, channels)
    size - The standard deviation of the gaussian for blurring in pixels, default=size of image / 6

    Outputs:
    ========
    new_ims - Stack of images of same number of dimensions as input but new sizes
    """
    if size == None:
        size = ims.shape[1] // 6
    if len(ims.shape) == 3:
        blur_vars = [size * np.sqrt(np.std(ims))]
    else:
        blur_vars = [size * np.sqrt(np.std(ims[..., k])) for k in range(ims.shape[-1])]
    new_ims = np.zeros_like(ims)
    if len(ims.shape) == 3:
        new_ims = gaussian_filter(ims, [0, blur_vars[0], blur_vars[0]])
    else:
        for k in range(ims.shape[-1]):
            new_ims[..., k] = gaussian_filter(ims[..., k], [0, blur_vars[k], blur_vars[k]])
    return new_ims

def rotate_stack(ims, theta=90):
    """
    Rotate all images on a stack the given theta

    Args:
    =====
    ims - Stack of images: 3d - (frames, height, wid) or 4d - (frames, height, wid, channels)
    theta - how much to rotate the image in degrees (only accepts 0, 90, 180, 270)

    Outputs:
    ========
    new_ims - Stack of images of same number of dimensions as input but new sizes
    """
    new_ims = np.rot90(ims, k=theta//90, axes=(1,2))
    return new_ims

def flip_stack(ims, direction="x"):
    """
    Flip all images on a stack according to given direction

    Args:
    =====
    ims - Stack of images: 3d - (frames, height, wid) or 4d - (frames, height, wid, channels)
    direction - the direction of flipping ("x" or "y")

    Outputs:
    ========
    new_ims - Stack of images of same number of dimensions as input but new sizes
    """
    if direction == "x":
        new_ims = np.flip(ims, axis=2)
    elif direction == "y":
        new_ims = np.flip(ims, axis=1)
    else:
        new_ims = ims

    return new_ims

def grayscale_stack(ims):
    """
    Change all images in a stack to 2d grayscale images

    Args:
    =====
    ims - Stack of images: 4d - (frames, height, wid, channels)

    Outputs:
    ========
    new_ims - Stack of images of same number of dimensions as input but new sizes
    """
    new_ims = []
    for im in ims:
        new_ims.append(rgb2gray(im))
    return np.array(new_ims)[..., np.newaxis]

def hsv_stack(ims):
    """
    Change all images in a stack to hsv images

    Args:
    =====
    ims - Stack of images: 4d - (frames, height, wid, channels)

        Outputs:
        ========
    new_ims - Stack of images of same number of dimensions as input but new sizes
    """
    new_ims = []
    for im in ims:
        new_ims.append(rgb2hsv(im))
    return np.array(new_ims)
