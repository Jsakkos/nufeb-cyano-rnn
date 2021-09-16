import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.segmentation import watershed

# Colony level comparison metrics
#
# Functions should accept sequences of bacteria images and extract metrics
# Sequences of the form:
#               frames x height x width x channels

def threshold_ims(ims, color=None):
    """ Threshold images in stack with Otsu. If 3 channels, treat each separately """
    t_ims = grab_color(ims, color)
    thresh_ims = np.zeros_like(t_ims)
    if thresh_ims.ndim == 4:
        for i in range(thresh_ims.shape[-1]):
            thresh = threshold_otsu(t_ims[..., i])
            if thresh >= 1e-5:
                mask = t_ims[..., i] >= thresh
                thresh_ims[mask, i] = 1
    else:
        thresh = threshold_otsu(t_ims)
        thresh_ims[t_ims >= thresh] = 1
    return thresh_ims

def threshold_im(im, color=None):
    """ Threshold single image with Otsu. If 3 channels, treat each separately """
    t_im = grab_color(im, color)
    thresh_im = np.zeros_like(t_im)
    if thresh_im.ndim == 3:
        for i in range(thresh_im.shape[-1]):
            thresh = threshold_otsu(t_im[..., i])
            if thresh >= 1e-5:
                mask = t_im[..., i] >= thresh
                thresh_im[mask, i] = 1
    else:
        thresh = threshold_otsu(t_im)
        thresh_im[t_im >= thresh] = 1
    return thresh_im

def grab_color(ims, color=None):
    """ Grab R,G,or B from RGB image or return original if None """
    if color == "red":
        t_ims = ims[..., 0]
    elif color == "green":
        t_ims = ims[..., 1]
    elif color == "blue":
        t_ims = ims[..., 2]
    else:
        t_ims = ims
    return t_ims

def extract_population(ims, color=None):
    """ Sum pixel intensities of given color or all colors if color=None"""
    t_ims = grab_color(ims, color)
    pop = np.sum(t_ims, axis=tuple(range(1, t_ims.ndim)))
    return pop

def extract_thresh_population(ims, color=None):
    """ Sum Otsu thresholded pixel intensities of given color """
    t_ims = grab_color(ims, color)
    thresh_ims = threshold_ims(t_ims)
    thresh_pop = np.sum(thresh_ims, axis=tuple(range(1, thresh_ims.ndim)))
    return thresh_pop

def extract_colony_labels(im, color=None, min_area=3):
    """ Threshold single image and label colonies of each color """
    t_im = grab_color(im, color)
    thresh_im = threshold_im(t_im)
    labels = measure.label(thresh_im, connectivity=2)
    for i in np.unique(labels):
        if i != 0:
            mask = thresh_im == i
            area = np.sum(mask)
            if area < min_area:
                labels[mask] = 0
    return labels

def extract_colony_locs(im, color=None):
    """ Get centroid of colonies of given color in image """
    t_im = grab_color(im, color)
    labels = extract_colony_labels(t_im)
    centers = [spot["centroid"] for spot in measure.regionprops(labels)]
    return np.array(centers)

def extract_colony_areas(im, color=None):
    """ Get areas of colonies of given color in image """
    t_im = grab_color(im, color)
    labels = extract_colony_labels(t_im)
    areas = [spot["area"] for spot in measure.regionprops(labels)]
    return np.array(areas)

def extract_num_colonies(im, color=None):
    """ Get number of colonies of given color in image """
    t_im = grab_color(im, color)
    labels = extract_colony_labels(t_im)
    return len(np.unique(labels)) - 1

def track_colonies(ims, color=None, method="watershed"):
    """ Identify starting seed colonies and label over time """
    t_ims = threshold_ims(ims, color)
    t_labels = np.zeros(t_ims.shape).astype(int)
    Y, X = np.indices(t_ims.shape[1:3])

    if method == "watershed":
        """ Fill in thresholded pixels with original colony labels """
        if t_ims.ndim == 3:
            t_labels[0] = measure.label(t_ims[0], connectivity=2)
            t_labels[0] = extract_colony_labels(t_ims[0])
            for k in range(1, t_ims.shape[0]):
                # Fill out the labels from the previous step to any thresholded pixels
                t_labels[k] = watershed(t_ims[k], t_labels[k-1], 2, mask=t_ims[k])
        else:
            for channel in range(t_ims.shape[-1]):
                t_labels[0, ..., channel] = measure.label(t_ims[0, ..., channel], connectivity=2)
            for k in range(1, t_ims.shape[0]):
                # Fill out the labels from the previous step to any thresholded pixels
                for channel in range(t_ims.shape[-1]):
                    t_labels[k, ..., channel] = watershed(t_ims[k, ..., channel], t_labels[k-1, ..., channel], 2, mask=t_ims[k, ..., channel])
    elif method == "distance":
        """ Take closest pixels to colony centers, make colony, then update centers """
        if t_ims.ndim == 3:
            t_labels[0] = measure.label(t_ims[0], connectivity=2)
            centers = np.array([colony["centroid"] for colony in measure.regionprops(t_labels[0])])
            for k in range(1, t_ims.shape[0]):
                for i in range(t_ims.shape[1]):
                    for j in range(t_ims.shape[2]):
                        if t_ims[k, i, j] != 0:
                            dists = np.linalg.norm(np.array([j, i]) - centers, axis=1)
                            closest = np.argmin(dists)
                            t_labels[k, i, j] = closest+1
                # Update centers
                for ci in range(centers.shape[0]):
                    mask = t_labels[k] == ci+1
                    col_size = np.sum(mask)
                    if col_size > 0:
                        centers[ci, 0] = np.mean(X[mask])
                        centers[ci, 1] = np.mean(Y[mask])
        else:
            for channel in range(t_ims.shape[-1]):
                t_labels[0, ..., channel] = measure.label(t_ims[0, ..., channel], connectivity=2)
                centers = np.array([colony["centroid"] for colony in measure.regionprops(t_labels[0, ..., channel])])
                for k in range(1, t_ims.shape[0]):
                    for i in range(t_ims.shape[1]):
                        for j in range(t_ims.shape[2]):
                            if t_ims[k, i, j, channel] != 0:
                                dists = np.linalg.norm(np.array([j, i]) - centers, axis=1)
                                closest = np.argmin(dists)
                                t_labels[k, i, j, channel] = closest+1
                    # Update centers
                    for ci in range(centers.shape[0]):
                        mask = t_labels[k, ..., channel] == ci+1
                        col_size = np.sum(mask)
                        if col_size > 0:
                            centers[ci, 0] = np.mean(X[mask])
                            centers[ci, 1] = np.mean(Y[mask])

    else:
        print("Tracking colonies: Unknown method")

    return t_labels
