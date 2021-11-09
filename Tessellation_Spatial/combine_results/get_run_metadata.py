import os
import sys
import re


fulldir = sys.argv[1]#"/home/joe/professional/research/NUFEB-cyanobacteria/data/exploratory/fourth_test_with_dist/distributions/Run_83_1"
runs_dir = os.path.dirname(fulldir)#"/home/joe/professional/research/NUFEB-cyanobacteria/data/exploratory/fourth_test_with_dist/distributions"
run_dir = os.path.basename(fulldir)#"Run_83_1"
run_code = run_dir[4:]


run_name = f'Run_{run_code}'
write_dir = os.path.join(runs_dir,run_name,'Results','shape_metrics')
atom_file = os.path.join(runs_dir,f'atom_{run_code}.in')
run_dir =os.path.join(runs_dir, run_name)

HEADER = 0
ATOMS = 1
NUTRIENTS = 2
TYPES = 3
DIFFUSION = 4
GROWTH = 5
KS = 6
BIO_YIELD = 7
MAINTENANCE = 8
DECAY = 9
read_state = HEADER

spatial_pattern = ''
spatial_intensity = ''
spatial_radius = ''
spatial_cluster_pts = ''
atoms = {}
print(f'Reading {atom_file}')
with open(atom_file,'r') as f:
    for line in f:
        l = line.rstrip()
        if(len(l)>0):
            if(HEADER == read_state):
                spat_patt_match = re.search('# Pattern: (.*)',l)
                spat_intensity_match = re.search('# Realized Intensity: (.*)',l)
                spat_radial_match = re.search('# Radius: (.*)',l)
                spat_cluster_match = re.search('# Pts. Per Cluster: (.*)',l)
                if(spat_patt_match):
                    spatial_pattern = spat_patt_match.group(1)
                if(spat_intensity_match):
                    spatial_intensity = spat_intensity_match.group(1)
                if(spat_radial_match):
                    spatial_radius = spat_radial_match.group(1)
                if(spat_cluster_match):
                    spatial_cluster_pts = spat_cluster_match.group(1)
                if(re.search('Atoms$',l)):
                   read_state = ATOMS
                   print('Reading Atoms data')
            elif(ATOMS == read_state):
                #      1 1 1.82e-06  370 4.11e-05 8.73e-05 1.00e-09 1.82e-06
                atom_matcher = re.search('\s*([0-9]+) [0-9]+ ([0-9]*\.[0-9]*e-*[0-9]*)',l)
                if(atom_matcher):
                    atoms[atom_matcher.group(1)]=atom_matcher.group(2)
                if(re.search('Nutrients$',l)):
                   read_state = NUTRIENTS
                   print('Got to Nutrient data')
            elif(ATOMS == read_state):
                break;
with open(os.path.join(write_dir,'spatial_distribution.csv'),'w') as f:
    f.write('RunID,Pattern, Intensity, Radius, Points Per Cluster\n')
    f.write(f'{run_name},{spatial_pattern},{spatial_intensity},{spatial_radius},{spatial_cluster_pts}\n')


with open(os.path.join(write_dir,'atom_sizes.csv'),'w') as f:
    f.write('RunID,Atom,Diameter (meters)\n')
    for k,v in atoms.items():
        f.write(f'{run_name},{k},{v}\n')
