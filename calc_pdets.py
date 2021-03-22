import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import os
import time
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial

from astropy.cosmology import z_at_value
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from utils import eccentric_selection_utils


# --- Argument handling --- #
argp = argparse.ArgumentParser()
argp.add_argument("--datapath", type=str, required=True, help="Path to where the population models are living. Models should contain masses, spins, redshifts, and eccentricities at some reference frequency.")
argp.add_argument("--outpath", type=str, required=True, help="Output path for the dataframe.")
argp.add_argument("--psdpath", type=str, required=True, help="Path to directory with PSD files, saved in same format as Observing Scenarios data release.")
argp.add_argument("--sensitivity", type=str, default="midhighlatelow", help="LIGO/Virgo sensitivity to use from LVC observing scenarios. Default='midhighlatelow'.")
argp.add_argument("--ifos", nargs="+", default=["H1"], help="Specifies which interferometers, or network of interferometers, to consider when calculating detection probabilities. Default='H1'.")
argp.add_argument("--fLow", type=float, default=10.0, help="Low frequency sensitivity, should be consistent with the eccentricities that are calculated in the population data. Default=10.0.")
argp.add_argument("--snr-thresh", type=float, default=8.0, help="SNR threshold for detection. If more than one detector is specified in args.ifos, this will be the network SNR threshold. Default=8.0.")
argp.add_argument("--Ntrials", type=int, default=1000, help="Number of trials to perform when calculating detection probability. Default=1000.")
argp.add_argument("--num-cores", type=int, default=1, help="Sets the number of cores to parallelize the calculation over. By default, will just run on a single core.")
args = argp.parse_args()


### Read in cluster data
data = pd.read_hdf(args.datapath, key='bbh')

### Create dict for ifos and sensitivity
ifos_dict = {}
for ifo in args.ifos:
    ifos_dict[ifo] = args.sensitivity

### Calculate optimal matched-filter SNRs and detection probabilities for all systems
ecc_key = 'e_'+str(int(args.fLow))+'Hz'
if ecc_key not in data.keys():
    raise ValueError("Eccentricities are not calculated for your specified fLow of {:0.1}Hz".format(args.fLow))
pdet_data = np.asarray(data[['m1','m2','spin1','spin2','luminosity_distance',ecc_key]])

func = partial(eccentric_selection_utils.detection_probability_eccentric, psd_path=args.psdpath, ifos=ifos_dict, snr_thresh=args.snr_thresh, Ntrials=args.Ntrials, aligned_spin=False, f_low=args.fLow)
pool = multiprocessing.Pool(args.num_cores)

start=time.time()
results = pool.map(func, pdet_data)
end=time.time()
print("Detection probability calculations took {:0.3} seconds".format(end-start))
results = np.asarray(results)   # [Nsys x 3 x 3]

### Add results to population dataframe
for idx, key in enumerate(['snr_opt_cc','snr_opt_ee','snr_opt_ce']):
    data[key] = np.asarray(results[:,0,idx])
for idx, key in enumerate(['chi2_cc','chi2_ee','chi2_ce']):
    data[key] = np.asarray(results[:,1,idx])
for idx, key in enumerate(['pdet_cc','pdet_ee','pdet_ce']):
    data[key] = np.asarray(results[:,2,idx])

data.to_hdf(args.outpath, key='bbh')



