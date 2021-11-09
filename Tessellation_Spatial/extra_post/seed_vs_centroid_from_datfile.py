# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Jul 15 14:08:38 2021

# @author: joe
# """

#     seeds_and_facets = associate_seeds_facets(seeds, clipped_voronoi)
#     df = pd.DataFrame([[k,
#                         cv2.contourArea(v[0]),
#                         cv2.arcLength(v[0], True),
#                         facet_AR(v[0]),
#                         len(v[0])-1,
#                         v[1]] for k, v in seeds_and_facets.items()],
#                       columns=['Seed ID', 'Facet Area (pixels)',
#                                'Facet Perimeter (pixels)',
#                                'Facet Aspect Ratio', 'Facet Sides',
#                                'Is Edge Facet'])

#     df['Scale (pixels per meter)'] = scale


# CELL_ANCESTOR_COL = 5

# # Magic value indicating a cell has not yet been assigned an ancestor
# NO_ANCESTOR_ASSIGNED = -1


# get the raw voroni facets
import seaborn as sns
from scipy import ndimage  # image manipulation
import colorsys  # color code managment
from matplotlib import path  # dealing with voronoi facets polylines as paths
import cv2
import pandas as pd  # Data frames from numpy arrays, especially for output
import numpy as np  # operate on data from the hdf5 file and image generation
import os


def calc_voronoi_from_seeds(height, width, seeds):
    """
    Generate the voronoi facets for an an experiment area, based on the seeding
    cell locations.

    Parameters
    ----------
    height : Height in pixels of the experimental area.
    seeds : A numpy array conforming the to return value of :func: get_seeds
        Specifically, each row corresponds to one seed and the columnts at
        CELL_X_COL and CELL_Y_COL give the X and Y coordinates in pixel units

    Returns
    -------
    A list of all Voronoi facets, where each facet is an array of vertices
    corresponding to pixel locations.

    """
    # prepare the subdivsion area
    rect = (0, 0, height, width)
    subdiv = cv2.Subdiv2D(rect)

    # load all seed locations into subdiv as (x,y) tuples
    # TODO probably a more efficient way.  convert np columns directly to
    # vector of tuples?
    for s in seeds[:, [CELL_X_COL, CELL_Y_COL]]:
        subdiv.insert((s[0], s[1]))
    (facets, centers) = subdiv.getVoronoiFacetList([])
    return(facets)


# %%

# %%
# clip the facets to the bounding box and trim to ints
# doesn't just bound existing vertices, also creates new along bounds
# using matplotlib path clipping to handle the math

# TODO change this to require seed_facets, so that we can track seed_ids with
# clipped facets (e.g. edge colonies)


def clip_facets(facets, bound_height, bound_width):
    """
    Clips facets to a given bounding area and rounds to the nearest
    integer.

    Parameters
    ----------
    facets : A list of all Voronoi facets, where each facet is an array of
    vertices
    corresponding to pixel locations.
    bound_height : Height of the bounding area, same units as those in the
    facet list.
    bound_width : Height of the bounding area, same units as those in the
    facet list

    Returns
    -------
    A list of clipped Voronoi facets, where each facet is an array of vertices
    corresponding to pixel locations.

    """
    clipped_facets = []
    ifacets = []
    rect = (0, 0, bound_height, bound_width)
    for fi, f in enumerate(facets):
        mpp = path.Path(f, closed=True)
        edge_piece = False
        for vert in f:
            vert_under_x = (vert[0] <= rect[0])
            vert_over_x = (vert[0] >= rect[3])
            vert_under_y = (vert[1] <= rect[1])
            vert_over_y = (vert[1] >= rect[2])
            edge_piece = (edge_piece | vert_under_x | vert_over_x |
                          vert_under_y | vert_over_y)
        clipped = mpp.clip_to_bbox(rect)
        clipped_facets.append(clipped)
        point_arr = []
        for points, code in clipped.iter_segments():
            point_arr.append(points)
        pa = np.array(point_arr, np.int)
        ifacets.append([pa, edge_piece])
    return(ifacets)


# create a dictionary associated seed points (by id) with each facet
def associate_seeds_facets(seeds, facets):
    # TODO docstring
    seed_facets = {}
    for seed in seeds:
        p = (seed[CELL_X_COL], seed[CELL_Y_COL], seed[CELL_Z_COL])
        for fi, f in enumerate(facets):
            mpp = path.Path(f[0], closed=True)
            if(mpp.contains_point(p)):
                seed_facets[seed[CELL_ID_COL]] = f
    return(seed_facets)


# def facet_AR(facet):
#     # TODO docstring
#     (x, y), (width, height), angle = cv2.minAreaRect(facet)
#     return(max(width, height)/min(width, height))


# internal data manipulations

# image generation, manipulation, and analysis

# plotting

# simple debug printing, enabled with DEBUG_PRINT
DEBUG_PRINT = False


# probably want to move to a more formal logging at some point
def dprint(s):
    if(DEBUG_PRINT):
        print(s)


np.random.seed(seed=1979)
CELL_ID_COL = 0
CELL_X_COL = 1
CELL_Y_COL = 2
CELL_Z_COL = 3
CELL_RADIUS_COL = 4
CELL_ANCESTOR_COL = 5

seed_data = pd.read_csv('2021-07-06-11-17-1625566660_point_patterns.csv')
grouped = seed_data.groupby('id')
pxpm = 5e6
run_seeds = {}
ofile = open('centroid_dist.csv', 'w')
ofile.write('RunID,seed,dist_m,centroid_x_m,centroid_y_m,seed_x_m,seed_y_m\n')
for name, group in grouped:
    xloc = group.x.to_numpy()*pxpm
    yloc = group.y.to_numpy()*pxpm
    seed_idx = np.arange(1, len(group)+1)
    zloc = np.zeros(len(group))
    radii = np.zeros(len(group))+1e-6
    seeds_genned = np.column_stack(
        (seed_idx, xloc, yloc, zloc, radii, seed_idx))

    seeds = seeds_genned

    height = 500
    width = 500
    raw_voronoi = calc_voronoi_from_seeds(height, width, seeds)
    raw_voronoi = calc_voronoi_from_seeds(height, width, seeds)
    clipped_voronoi = clip_facets(raw_voronoi, height, width)
    seeds_and_facets = associate_seeds_facets(seeds, clipped_voronoi)

    viz_scale = 1
    blank_image2 = np.ones((height*viz_scale,
                            width*viz_scale, 3), np.uint8)
    for k in seeds_and_facets.keys():
        # print(k)
        cnt = seeds_and_facets[k][0]
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        seed = seeds[seeds[:, CELL_ID_COL] == k]
        # print(seed)
        sX = seed[:, CELL_X_COL]
        sY = seed[:, CELL_Y_COL]
        dX = cX-sX
        dY = cY-sY
        dist = np.sqrt((dX*dX)+(dY*dY))
        dist_m = dist/pxpm
        # print(
        #    f'Run{name} Seed No:{k} X:{sX} Y:{sY}    Centroid X:{cX} Y:{cY}    Distance:{dist}pm {dist_m}m')
        ofile.write(
            f'Run_{name}_1,{int(k)},{dist_m[0]},{cX/pxpm},{cY/pxpm},{sX[0]/pxpm},{sY[0]/pxpm}\n')
        cv2.drawContours(blank_image2, [cnt]*viz_scale, -1, (122, 255, 0), 2)
        txt = f'{dist}'
        text_x_pos = int(sX)+2
        text_y_pos = int(sY)+2
        cv2.putText(blank_image2, txt, (text_x_pos, text_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 122, 255), 1)
        cv2.circle(blank_image2, (int(sX), int(sY)), 10, (0, 0, 255), -1)
        cv2.circle(blank_image2, (int(cX), int(cY)), 10, (255, 0, 0), -1)
        cv2.line(blank_image2, (int(sX), int(sY)),
                 (int(cX), int(cY)), (255, 255, 255))
   # cv2.imshow("Image", blank_image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.destroyAllWindows()
ofile.close()
