# internal data manipulations
import numpy as np  # operate on data from the hdf5 file and image generation
import pandas as pd  # Data frames from numpy arrays, especially for output

# image generation, manipulation, and analysis
import cv2
from matplotlib import path  # dealing with voronoi facets polylines as paths
import colorsys  # color code managment
from scipy import ndimage  # image manipulation

# plotting
import seaborn as sns

# simple debug printing, enabled with DEBUG_PRINT
DEBUG_PRINT = False


# probably want to move to a more formal logging at some point
def dprint(s):
    if(DEBUG_PRINT):
        print(s)


def get_timesteps(trajectory):
    """
    Determine valid timesteps for the associated hdf5 dump

    Parameters
    ----------
    trajectory : The decoded hdf5 file containing the dumped run data.
        Usually from something like h5py.File(infile, 'r')

    Returns
    -------
    A tuple consisting of three entries. The first is a sorted list of
    integers for every valid timestep.
    The second and third are for convenience and represent the start and
    end time of the run. (First and last timesteps.)

    """
    trajectory_times = sorted([int(k) for k in trajectory['id'].keys()])
    start_time = trajectory_times[0]
    end_time = trajectory_times[len(trajectory_times)-1]
    return(trajectory_times, start_time, end_time)


# Using numpy arrays for a lot of the work rather than pandas
# Setting up constants to keep column indices correct
CELL_ID_COL = 0
CELL_X_COL = 1
CELL_Y_COL = 2
CELL_Z_COL = 3
CELL_RADIUS_COL = 4
CELL_ANCESTOR_COL = 5

# Magic value indicating a cell has not yet been assigned an ancestor
NO_ANCESTOR_ASSIGNED = -1

# TODO, in a few places, it may be better to move to a more object oriented
# approach. Mainly for a more clear interface, while avoiding state. For
# example, get_timesteps could easily produce an object with 'steps()',
# 'start_time()', and 'end_time()' methods for clarity without affecting
# performance or relying on too much internal state.


# TODO make compatible with the newer, corrected hdf5 radius dump
def radius_key(timestep):
    """
    Generate the appropriate key for a radius at a given timestep.
    Does not check timestep for validity.

    This function exists because current phototroph runs use an older version
    of NUFEB which output individual radius keys for each timestep, (e.g.
    radius0, radius100, etc) rather than a single radius entry indexed by
    timestep.

    Parameters
    ----------
    timestep : The numeric time step at which we want radius info


    Returns
    -------
    A string representing the key for the radius information at the given
    timestep

    """
    return(f'radius{timestep}')


# TODO error out gracefully if time does not exist
def get_cells(trajectory, time=0, scale=1E6):
    """
    Provide the scaled location and radius of all cells at a particular
    timestep, with each cell associted with a tag id which remains consistent
    between timesteps.

    Scaling is intended mainly to translate spatial coordinate to image
    pixel locations.

    Parameters
    ----------
    trajectory : The decoded hdf5 file containing the dumped run data.
        Usually from something like h5py.File(infile, 'r')
    time : An integer representing the timestep to query for cell locations.
        Most runs start at time 0, so this has been left as a
        default value.
    scale : A value by which to multiply the physical coordinates.
        The inteded goal to convert from spatial coordinates to pixel locations
        so scale is generally passed a number representing pixels per meter.
        The default value returns distance in terms of microns.
        WARNING: Because we are return an integer based numpy array, setting
        the scale low (as it may be tempting to set the scale to 1) would lead
        to most values being 0.

    Returns
    -------
    A five column, multi-row numpy array. Where the columns, in order, are:
        1. The consistent atom tag (id)
        2. The scaled x, y, and z coordinates of the cell
        3. The cell radius
        4. The cell's ancestors. This column is intended for later bookkeeping
        and is not populated here, beyond initializing to NO_ANCESTOR_ASSIGNED

    Each column can be referenced by the defined constants:
        CELL_ID_COL = 0
        CELL_X_COL = 1
        CELL_Y_COL = 2
        CELL_Z_COL = 3
        CELL_RADIUS_COL = 4
        CELL_ANCESTOR_COL = 5
    """

    time = str(time)
    ret_array = np.column_stack(
        (trajectory['id'][time],
         scale*np.column_stack((trajectory['x'][time],
                                trajectory['y'][time],
                                trajectory['z'][time],
                                trajectory[radius_key(time)])),
         np.full((len(trajectory['id'][time]), 1), NO_ANCESTOR_ASSIGNED)
         )).astype(int)
    # Occasionally a cell with id == 0 is saved, this is not a valid cell
    return( ret_array[ret_array[:,CELL_ID_COL]!= 0])


def get_seeds(trajectory, start_time=0, scale=1E6):
    """
    As with get_cells:
    Provide the scaled location and radius of all cells at a particular
    timestep, with each cell associted with a tag id which remains consistent
    between timesteps.

    HOWEVER: Also assigns the ancestor id to the same as the cell id, since
    these are the initial cells.

    Parameters
    ----------
    trajectory : The decoded hdf5 file containing the dumped run data.
        Usually from something like h5py.File(infile, 'r')
    start_time : An integer representing the initial timestep.
        Most runs start at time 0, so this has been left as a
        default value.
    scale : A value by which to multiply the physical coordinates.
        The inteded goal to convert from spatial coordinates to pixel locations
        so scale is generally passed a number representing pixels per meter.
        The default value returns distance in terms of microns.
        WARNING: Because we are return an integer based numpy array, setting
        the scale low (as it may be tempting to set the scale to 1) would lead
        to most values being 0.

    Returns
    -------
    A five column, multi-row numpy array. Where the columns, in order, are:
        1. The consistent atom tag (id)
        2. The scaled x, y, and z coordinates of the cell
        3. The cell radius
        4. The cell's ancestors. Unlike with get_cells, this column is
        populated. Specfically, it ought to match the value in the CELL_ID_COL
        since these are the initial seeds.

    Each column can be referenced by the defined constants:
        CELL_ID_COL = 0
        CELL_X_COL = 1
        CELL_Y_COL = 2
        CELL_Z_COL = 3
        CELL_RADIUS_COL = 4
        CELL_ANCESTOR_COL = 5
    """
    seeds = get_cells(trajectory, start_time, scale)
    # Since this is the first set of cells, they are their own ancestor
    seeds[:, CELL_ANCESTOR_COL] = seeds[:, CELL_ID_COL]
    return(seeds)


# %%
def assign_ancestry(trajectory):
    """
    Infer the ancestor of all cells during all timesteps.

    Since cell ancestors are not necessarily tracked, we have to infer them
    as we go. This method steps through each timestep, identifies cells with
    unknown ancestors, and assigns them an ancestor based on the nearest cell
    with a known/inferred ancestor.

    There are many other approaches, but this one has proven to be the least
    brittle in practice.  Do note however, that the accuracy of the inference
    will likely go down if the time between recorded timesteps is too large.

    Although this does a brute force nearest-neighbor search, it has not
    proven to take very long for the number of cells used in our current
    runs (order of 1000).  There are internal comments noting where
    optimizations could be made.

    Parameters
    ----------
    trajectory : The decoded hdf5 file containing the dumped run data.
        Usually from something like h5py.File(infile, 'r')

    Returns
    -------
    A dictionary mapping each cell present in the timestep to the id of its
    ancestor.

    """

    dprint('Infeerring cell ancestries')
    trajectory_times, start_time, end_time = get_timesteps(trajectory)

    # Do not need to scale these, since we only care about relative distances
    seeds = get_seeds(trajectory, start_time=start_time)

    # Dictionary which will hold associations between cell ids and ancestors
    ancestry = {}

    # All seeds have a known ancestry, themselves
    for seed in seeds:
        ancestry[seed[CELL_ID_COL]] = seed[CELL_ANCESTOR_COL]

    for time in trajectory_times:
        dprint(f'\tProcessing time: {time}')
        # Do not need to scale, we only care about relative distances
        cells = get_cells(trajectory, time=time)

        # for cells with known ancestors, set the appropriate value in the
        # ancestor column. Used to filter cell list for those with unknown
        # ancestors
        for cell_id, anc_id in ancestry.items():
            # Every once in a while a cell leaves the simulation, so make sure
            # it actually exists at this timestep
            if(len(cells[cells[:, CELL_ID_COL] == cell_id]) > 0):
                ancestor_found = cells[cells[:, CELL_ID_COL] == cell_id][0]
                ancestor_found[CELL_ANCESTOR_COL] = anc_id
                cells[cells[:, CELL_ID_COL] == cell_id] = ancestor_found

        # for all the cells with no currently known ancestor, find the
        # nearest cell with an ancestor and assign that ancestor to the
        # unknown cell
        #
        # TODO if this gets slow, use kdtree for neighbor search
        # could also use some pre-sorting and heuristics, e.g. keep cells list
        # sorted by x,y  it's highly unlikely any daughter cell is going to be
        # hundreds of pixels away from its parent, so don't need to search the
        # whole list and, likely, we ought to do something else anyway if it
        # is that far away
        no_ancestor_found = cells[cells[:, CELL_ANCESTOR_COL] == -1]

        for naf in no_ancestor_found:
            x_new = naf[CELL_X_COL]
            y_new = naf[CELL_Y_COL]
            z_new = naf[CELL_Z_COL]
            naf_id = naf[CELL_ID_COL]
            min_dist = -1
            nearest_ancestor = -1
            for cell_id, anc2_id in ancestry.items():
                ancestor_found = cells[cells[:, CELL_ID_COL] == cell_id]
                if(len(ancestor_found) > 0):
                    x_old = ancestor_found[0, CELL_X_COL]
                    y_old = ancestor_found[0, CELL_Y_COL]
                    z_old = ancestor_found[0, CELL_Z_COL]
                    distance = ((x_old-x_new)*(x_old-x_new)
                                + (y_old-y_new)*(y_old-y_new)
                                + (z_old-z_new)*(z_old-z_new))
                    if((min_dist == -1) | (distance < min_dist)):
                        min_dist = distance
                        nearest_ancestor = anc2_id

            # now that we've found the nearest neighbor cell with a known
            # ancestor, update the ancestry dictionary
            ancestry[naf_id] = nearest_ancestor
            ancestor_found = cells[cells[:, CELL_ID_COL] == cell_id][0]
            ancestor_found[CELL_ANCESTOR_COL] = nearest_ancestor
            # probably don't need to do this update since cells is
            # about to go out of scope
            cells[cells[:, CELL_ID_COL] == cell_id] = ancestor_found
    return(ancestry)


# TODO this family of functions really ought to have some responsiblities
# split. Basically, there's filtering which colonies we care about and there's
# determining the area(s) of the relevant colonies.  As a motivating example
# think about how separating out the filter responsiblity would ease a new
# use case of 'show me only the live heterotrophs while ignoring the
# cyanobacteria and eps components'
def get_colony_morphology_at_time(time, ancestor_id, ancestors, trajectory,
                                  scale, height, width):
    """
    Determine the apparent 2D area of a colony at a specific timestep. A
    colony is defined as all cells sharing a common ancestor. The 2D apparent
    area is the visible biomass looking from the top down. Every cell from
    the colony is projected to the x-y plane and is occulded by any non-colony
    colony cells above them.

    Internally, this function generates a virtual black and white image of
    the projected and occluded colony to determine the apparent area. The
    scale, height, and width parameters should be set so that the results
    are comparable to any associated micrographs from analagous wet-lab
    experiments.

    This function may be called on its own, but it is originally intended as
    the lowest level component of :func: get_colony_morphologies_at_times.

    Parameters
    ----------
    ancestor_id : The numeric id of the common ancestor to all colony members.
    ancestors : A dictionary mapping each cell present in the timestep to the
        id of its ancestor.
    trajectory : The decoded hdf5 file containing the dumped run data.
        Usually from something like h5py.File(infile, 'r')
    time : The numeric timestep of interest.
    scale : A value by which to multiply the physical coordinates.
        The inteded goal to convert from spatial coordinates to pixel locations
        so scale is generally passed a number representing pixels per meter.
    height : The height of the virtual image.
    width : The width of the virtual image.


    Returns
    -------
    A three-element list containg the timestep, ancestor id, and apparent 2D
    area. Although techinically an ancestor id, the second item can also be
    thought of as a colony id.


    """
    dprint(f'Getting morphology of colony {ancestor_id} at time {time}')
    cells = get_cells(trajectory, time, scale)
    mask = np.zeros((height, width, 3), dtype="uint8")
    sorted_array = cells[np.argsort(cells[:, CELL_Z_COL])]
    for cell in sorted_array:
        loc = (int(cell[CELL_X_COL]), int(cell[CELL_Y_COL]))
        cell_id = cell[CELL_ID_COL]
        seed_id = ancestors[cell_id]
        if(seed_id == ancestor_id):
            color = (255, 255, 55)
        else:
            color = (0, 0, 0)
        cv2.circle(mask, loc, int(cell[CELL_RADIUS_COL]), color, -1)
    # for area, we just count white pixels. no need for cv2
    area = np.count_nonzero(mask)
    return([time, ancestor_id, area])


# %%

def get_colony_morphologies_at_time(time, ancestors, trajectory, scale, height,
                                    width):
    """
    Determine the apparent 2D areas of all colonies at a specific timestep. A
    colony is defined as all cells sharing a common ancestor. The 2D apparent
    area is the visible biomass looking from the top down. Every cell from
    the colony is projected to the x-y plane and is occulded by any non-colony
    colony cells above them.

    Internally, this function relies on a virtual black and white image of
    the projected and occluded colony to determine the apparent area. The
    scale, height, and width parameters should be set so that the results
    are comparable to any associated micrographs from analagous wet-lab
    experiments.

    This function may be called on its own, but it is originally intended as
    a mid-level component of :func: get_colony_morphologies_at_times.

    Parameters
    ----------
    ancestors : A dictionary mapping each cell present in the timestep to the
        id of its ancestor.
    trajectory : The decoded hdf5 file containing the dumped run data.
        Usually from something like h5py.File(infile, 'r')
    time : The numeric timestep of interest.
    scale : A value by which to multiply the physical coordinates.
        The inteded goal to convert from spatial coordinates to pixel locations
        so scale is generally passed a number representing pixels per meter.
    height : The height of the virtual image.
    width : The width of the virtual image.


    Returns
    -------
    A list of three-element lists which describes all colonies present at the
    given timestep. Each three-element list contains the timestep, ancestor id,
    and apparent 2D area. Although techinically an ancestor id, the second item
    can also be thought of as a colony id.

    """

    # TODO although it's conceptually pleasing to defer to doing one colony at
    # a time, it means there is A LOT of extra total drawing calls. Almost
    # certainly more efficient to draw all colonies on one image and count
    # the number pixels with the right color code
    morphologies = []
    for vi, v in enumerate(set(ancestors.values())):
        morphologies.append(
            get_colony_morphology_at_time(time, v, ancestors, trajectory,
                                          scale, height, width))
    return(morphologies)


# %%
def get_colony_morphologies_at_times(times, ancestors, trajectory, scale,
                                     height, width):
    """
    Determine the apparent 2D areas of all colonies at the specified times.
    A colony is defined as all cells sharing a common ancestor. The 2D apparent
    area is the visible biomass looking from the top down. Every cell from
    the colony is projected to the x-y plane and is occulded by any non-colony
    colony cells above them.

    Internally, this function relies on a virtual black and white image of
    the projected and occluded colony to determine the apparent area. The
    scale, height, and width parameters should be set so that the results
    are comparable to any associated micrographs from analagous wet-lab
    experiments.

    This function is intended as the main entry point to getting all colony
    areas over all timesteps of the simulation. Note that it may take a while
    to run. The subordinate functions :func: get_colony_morphologies_at_time
    and :func: get_colony_morphology_at_time can be called directly and may
    be useful for either prototyping/debugging or for when only a subset of
    colony areas (such as the areas at the final timestep) are of interest.

    Parameters
    ----------
    times : A numeric list of all timesteps of interest.
    ancestors : A dictionary mapping each cell present in the timestep to the
        id of its ancestor.
    trajectory : The decoded hdf5 file containing the dumped run data.
        Usually from something like h5py.File(infile, 'r')
    scale : A value by which to multiply the physical coordinates.
        The inteded goal to convert from spatial coordinates to pixel locations
        so scale is generally passed a number representing pixels per meter.
    height : The height of the virtual image.
    width : The width of the virtual image.


    Returns
    -------
    A Pandas dataframe which describes all colonies present at the
    requested timesteps. Each row contains the the timestep,
    ancestor id, apparent 2D area in pixels, and a record of the scaling factor
    between pixels and meters. Although techinically an ancestor id,
    the second item can also be thought of as a colony id.

    We are returning a dataframe, which is unlike the finer grained related
    functions for getting colony morophologies. In all uses so far, the
    originally returned list of arrays was immediately converted to a dataframe
    so we are incorporating that step.

    If dealing with a raw numpy array is required, the returned dataframe
    may be converted usings the :func: Pandas.dataframe.to_numpy method

    """
    morphologies = []
    for time in times:
        morphologies.extend(
                get_colony_morphologies_at_time(time, ancestors, trajectory,
                                                scale, height, width))
    df = pd.DataFrame(morphologies,
                      columns=['Time (s)', 'Colony ID', 'Area (pixels)'])
    df['Scale (pixels per meter)'] = scale
    return(df)


# get the raw voroni facets
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


def facet_AR(facet):
    # TODO docstring
    (x, y), (width, height), angle = cv2.minAreaRect(facet)
    return(max(width, height)/min(width, height))


def get_facet_morphologies(trajectory, scale, height, width):
    # TODO docstring
    timesteps, start_time, end_time = get_timesteps(trajectory)
    seeds = get_seeds(trajectory, start_time=start_time, scale=scale)
    raw_voronoi = calc_voronoi_from_seeds(height, width, seeds)
    clipped_voronoi = clip_facets(raw_voronoi, height, width)
    seeds_and_facets = associate_seeds_facets(seeds, clipped_voronoi)
    df = pd.DataFrame([[k,
                        cv2.contourArea(v[0]),
                        cv2.arcLength(v[0], True),
                        facet_AR(v[0]),
                        len(v[0])-1,
                        v[1]] for k, v in seeds_and_facets.items()],
                      columns=['Seed ID', 'Facet Area (pixels)',
                               'Facet Perimeter (pixels)',
                               'Facet Aspect Ratio', 'Facet Sides',
                               'Is Edge Facet'])

    df['Scale (pixels per meter)'] = scale
    return(df)


def RGB_palette(n, s=0.5, v=0.5, scale=255):
    # TODO docstring
    N = n
    HSV_tuples = [(x*1.0/N, s, v) for x in range(N)]
    HSV_tuples[0]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    RGB_scaled = []
    for x in RGB_tuples:
        RGB_scaled.append((int(x[0]*255), int(x[1]*255), int(x[2]*255)))
    return(RGB_scaled)


def combine_morphologies(colonies, facets):
    # TODO docstring
    joined_alltime = colonies.set_index('Colony ID').join(
        facets.set_index('Seed ID'), rsuffix=' (facet)')
    joined_alltime['Seed ID'] = joined_alltime.index
    joined_alltime['Winner Index'] = (joined_alltime['Area (pixels)'] /
                                      joined_alltime['Facet Area (pixels)'])

    median_colony_area = joined_alltime.groupby('Time (s)')['Area (pixels)'] \
        .median().to_frame()
    median_colony_area.rename(columns={'Area (pixels)':
                                       'Median Colony Area (pixels)'},
                              inplace=True)
    ja = joined_alltime.set_index('Time (s)').join(
        median_colony_area)
    ja['Median-Normed Colony Area'] = ((ja['Area (pixels)']
                                       - ja['Median Colony Area (pixels)'])
                                       / ja['Median Colony Area (pixels)'])
    ja['Colony ID'] = ja['Seed ID']
    return(ja)


def plot_colony_growth(colony_morphologies):
    # TODO docstring
    plot = sns.scatterplot(x=colony_morphologies['Time (s)'],
                           y=colony_morphologies['Area (pixels)'],
                           hue=colony_morphologies['Colony ID'],
                           palette="colorblind")
    return(plot)


def graph_facets(trajectory, scale, input_height, input_width, viz_scale):
    # TODO docstring
    timesteps, start_time, end_time = get_timesteps(trajectory)
    seeds = get_seeds(trajectory, start_time=start_time, scale=scale)
    raw_voronoi = calc_voronoi_from_seeds(input_height, input_width, seeds)
    clipped_voronoi = clip_facets(raw_voronoi, input_height, input_width)
    sf = associate_seeds_facets(seeds, clipped_voronoi)

    blank_image2 = np.ones((input_height*viz_scale,
                            input_width*viz_scale, 3), np.uint8)
    for fi, (s, f) in enumerate(sf.items()):
        num_seeds = len(sf.keys())
        RGB_seeds = RGB_palette(num_seeds, v=0.9)
        RGB_seeds_border = RGB_palette(num_seeds, v=0.1)
        RGB_facets_non_edge = RGB_palette(num_seeds, s=0.7, v=0.5)
        RGB_facets_edge = RGB_palette(num_seeds, s=0.15, v=0.5)
        seed = seeds[np.where(seeds[:, CELL_ID_COL] == s)]
        loc = (int(seed[:, CELL_X_COL]*viz_scale),
               int(seed[:, CELL_Y_COL])*viz_scale)
        radius = int(seed[:, CELL_RADIUS_COL])*viz_scale
        points = [f[0]*viz_scale]

        # Coloring of facets is slightly different if they're edge pieces
        if(f[1]):
            cv2.fillPoly(blank_image2, pts=points, color=RGB_facets_edge[fi-1])
            cv2.polylines(blank_image2, pts=points, color=(0, 0, 0),
                          isClosed=True, thickness=2)
        if(not f[1]):
            cv2.fillPoly(blank_image2, pts=points,
                         color=RGB_facets_non_edge[fi-1])
            cv2.polylines(blank_image2, pts=points, color=(0, 0, 0),
                          isClosed=True, thickness=3)
        cv2.circle(blank_image2, loc, max(radius+5, int(radius*1.5)),
                   RGB_seeds_border[fi-1], -1)
        seed_color = RGB_seeds[fi-1]
        cv2.circle(blank_image2, loc, radius, seed_color, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(blank_image2, str(s),
                    (int(seed[:, CELL_X_COL])*viz_scale-int(radius/2),
                     int(seed[:, CELL_Y_COL])*viz_scale-int(radius*2)),
                    font, 1.3, (255, 255, 255), 2, cv2.LINE_AA, False)
    return(blank_image2)


def plot_colonies_at_time(timestep, trajectory, scale, ancestry, input_height,
                          input_width, viz_scale):
    # TODO docstring

    # should probably just send the number of seeds
    num_seeds = len(set(ancestry.values()))
    N = num_seeds

    # TODO use the RBG_palette function
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    HSV_tuples[0]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]

    # TODO validate timestep
    # might be better to just send it a list of cells
    cells = get_cells(trajectory, timestep, scale)

    mask = np.zeros((input_height, input_width, 3))
    sorted_array = cells[np.argsort(cells[:, CELL_Z_COL])]
    for colony in sorted_array:
        loc = (int(colony[CELL_X_COL]), int(colony[CELL_Y_COL]))
        seed_id = ancestry[colony[CELL_ID_COL]]
        color = RGB_tuples[seed_id-1]
        cv2.circle(mask, loc, int(colony[CELL_RADIUS_COL]), color, -1)

    # TODO draw colony labels, perhaps optionally

    rotated_img = ndimage.rotate(mask, 90)
    # TODO handle axis flipping internally
    return(rotated_img)


if __name__ == "__main__":
    print("This file is not intened to be called directly.")
