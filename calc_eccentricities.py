import numpy as np
import pandas as pd

import os
import time
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial

import astropy.units as u
from astropy.cosmology import Planck15 as cosmo

from utils import gw_inspiral


# --- Argument handling --- #
argp = argparse.ArgumentParser()
argp.add_argument("--datapath", type=str, required=True, help="Path to where cluster  models are living. Data is stored in hdf5 file under key 'bbh'. Data should contain at least keys 'z_mergers', 'm1', 'm2', 'a_final(AU)', 'e_final'.")
argp.add_argument("--outpath", type=str, required=True, help="Output path for the dataframe.")
argp.add_argument("--freqs", nargs="+", default=[10], help="Specifies which frequencies to calculate eccentricities at. Default=[10].")
argp.add_argument("--num-cores", type=int, default=1, help="Sets the number of cores to parallelize the calculation over. By default, will just run on a single core.")
args = argp.parse_args()


### Read in cluster data
data = pd.read_hdf(args.datapath, key='bbh')
#Merger channels: 1)Ejected 2)In-cluster 2-body 3)In-cluster 3-body 4)In-cluster 4-body 5)In-cluster single-single capture

# remove systems that don't merge within a Hubble time
data = data.loc[~data['z_mergers'].isna()]

# get luminosity and comoving distance of mergers (Gpc)
data['luminosity_distance'] = cosmo.luminosity_distance(data['z_mergers']).to(u.Mpc).value
data['comoving_distance'] = cosmo.comoving_distance(data['z_mergers']).to(u.Mpc).value

# get cosmological weights
data['cosmo_weight'] = cosmo.differential_comoving_volume(data['z_mergers']) * (1+data['z_mergers'])**(-1)


### Calculate eccentricities at each frequency
for fLow in args.freqs:
    print('Calculating eccentricities at frequency of {} Hz...'.format(fLow))
    fLow = float(fLow)

    system_data = np.asarray(data[['m1','m2','a_final(AU)','e_final']])

    func = partial(gw_inspiral.eccentricity_at_ref_freq, f_gw=fLow, e_ref=0.999, R_peri=False)
    pool = multiprocessing.Pool(args.num_cores)

    start=time.time()
    results = pool.map(func, system_data)
    end=time.time()
    print("   eccentricity calculations took {:0.3} seconds\n".format(end-start))

    ecc_key = 'e_'+str(int(fLow))+'Hz'
    data[ecc_key] = np.asarray(results)


### Save processed cluster data
data.to_hdf(args.outpath, key='bbh')
