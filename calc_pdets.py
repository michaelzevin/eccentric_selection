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
argp.add_argument("--flow", type=float, default=10.0, help="Low frequency sensitivity, should be consistent with the eccentricities that are calculated in the population data. Default=10.0.")
argp.add_argument("--samp-rate", type=float, default=1024, help="Sampling rate for waveform generator, which affects timesliding to maximize SNR. Default=1024.")
argp.add_argument("--snr-thresh", type=float, default=8.0, help="SNR threshold for detection. If more than one detector is specified in args.ifos, this will be the network SNR threshold. Default=8.0.")
argp.add_argument("--Ntrials", type=int, default=1000, help="Number of trials to perform when calculating detection probability. Default=1000.")
argp.add_argument("--num-cores", type=int, default=1, help="Sets the number of cores to parallelize the calculation over. By default, will just run on a single core.")
args = argp.parse_args()


### Read in CMC data
data = pd.read_csv(args.datapath, sep=' ', skiprows=[0,1], \
                  names=['model','rv','Rgc','Z','N(x10^5)','t_merge(Myr)','id1','id2','m1','m2','spin1','spin2',\
                         'v_kick','v_esc','channel','id_merger','m_merger','spin_merger','a_final(AU)',\
                         'e_final','e_10Hz','cluster_age(Myr)','t_merger_actual(Myr)'])
# synthesize lookback time
data['t_merger_lookback(Myr)'] =  13700 - data['t_merger_actual(Myr)']
# get redshifts by interpolation
lookback_grid = np.linspace(0.01,13700, 1000) * u.Myr
z_grid = [z_at_value(cosmo.lookback_time, l) for l in lookback_grid]
lookback_to_redshift_interp = interp1d(lookback_grid, z_grid)
data['redshift'] = lookback_to_redshift_interp(data['t_merger_lookback(Myr)'])
data['dL'] = cosmo.luminosity_distance(data['redshift']).to(u.Mpc).value
# get cosmological weights
data['cosmo_weight'] = cosmo.differential_comoving_volume(data['redshift']) * (1+data['redshift'])**(-1)


### Create dict for ifos and sensitivity
ifos_dict = {}
for ifo in args.ifos:
    ifos_dict[ifo] = args.sensitivity


### Calculate optimal matched-filter SNRs and detection probabilities for all systems
pdet_data = np.asarray(data[['m1','m2','spin1','spin2','dL','e_10Hz']])

func = partial(eccentric_selection_utils.detection_probability_eccentric, psd_path=args.psdpath, ifos=ifos_dict, snr_thresh=args.snr_thresh, Ntrials=args.Ntrials, aligned_spin=False, flow=args.flow)
pool = multiprocessing.Pool(args.num_cores)

start=time.time()
results = pool.map(func, pdet_data)
end=time.time()
print("Detection probability calculations took {:0.3} seconds".format(end-start))


### Add results to population dataframe
snr_opt_cc, snr_opt_ee, snr_opt_ce, pdet_cc, pdet_ee, pdet_ce = [], [], [], [], [], []
for out in results:
    snr_opt_cc.append(out[0][0])
    snr_opt_ee.append(out[0][1])
    snr_opt_ce.append(out[0][2])
    pdet_cc.append(out[1][0])
    pdet_ee.append(out[1][1])
    pdet_ce.append(out[1][2])

data['snr_opt_cc'] = snr_opt_cc
data['snr_opt_ee'] = snr_opt_ee
data['snr_opt_ce'] = snr_opt_ce
data['pdet_cc'] = pdet_cc
data['pdet_ee'] = pdet_ee
data['pdet_ce'] = pdet_ce

data.to_csv(args.outpath, index=False, sep=',')



