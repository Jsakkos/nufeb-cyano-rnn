import pandas as pd  # Data frames from numpy arrays, especially for output
import numpy as np  # operate on data from the hdf5 file and image generation
import scipy.spatial.distance as dist

seed_data = pd.read_csv('2021-07-06-11-17-1625566660_point_patterns.csv')
grouped = seed_data.groupby('id')

run_seeds = {}
ofile = open('seed_dist_metrics.csv', 'w')
ofile.write('RunID,seed,log_inv_sq_dist_m,seed_x_m,seed_y_m\n')
for name, group in grouped:
    xloc = group.x.to_numpy()
    yloc = group.y.to_numpy()
    seed_idx = np.arange(1, len(group)+1)
    zloc = np.zeros(len(group))
    radii = np.zeros(len(group))+1e-6
    seeds_genned = np.column_stack(
        (seed_idx, xloc, yloc, zloc, radii, seed_idx))
    seeds = seeds_genned
    seed_xy = np.column_stack((xloc, yloc))
    cd = dist.pdist(seed_xy)
    #m * i + j - ((i + 2) * (i + 1)) // 2
    md = dist.squareform(cd)
    nsq = np.square(md)
    rowsum = np.sum(nsq, axis=0)
    loginvsqdist = np.log(rowsum)
    for i, seed in enumerate(seeds_genned):
        rowstr = f'Run_{name}_1,{int(seed[0])},{loginvsqdist[i]},{seed[1]},{seed[2]}\n'
        print(rowstr)
        ofile.write(rowstr)

ofile.close()
