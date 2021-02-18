import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os

from pycbc.detector import Detector

import EOBRun_module


# Waveform generation
def modes_to_k(modes):
    """
    Map (l,m) -> k
    """
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

def JBJF(hp,hc,dt):
    """
    Fourier transform of TD wfv
    """
    hptilde = np.fft.rfft(hp) * dt
    hctilde = np.fft.rfft(-hc) * dt
    return hptilde, hctilde

def gen_eccentric_waveform(M1, M2, Chi1, Chi2, Deff, e0, Iota=np.pi/2, modes=[[2,2]], f0=10, srate=1024):
    """
    Generates (time-domain) eccentric waveform using TEOBResumS
    
    Returns time, h_plus, h_cross
    """
    
    k = modes_to_k(modes)
    
    pars = {
        'M'                        : M1+M2,
        'q'                        : M2/M1,
        'Lambda1'                  : 0,
        'Lambda2'                  : 0,
        'chi1'                     : Chi1,
        'chi2'                     : Chi2,
        'ecc'                      : e0,
        'distance'                 : Deff,    # Mpc
        'inclination'              : Iota,
        'domain'                   : 0,       # TD
        'arg_out'                  : 0,       #Output hlm/hflm. Default = 0
        'use_mode_lm'              : k,
        'output_lm'                : k,       #List of modes to print on file
        'srate_interp'             : srate,
        'use_geometric_units'      : 0,
        'initial_frequency'        : f0,
        'interp_uniform_grid'      : 2,
        'df'                       : 0.01,    #df for FD interpolation
    }
    
    t, hp, hc = EOBRun_module.EOBRunPy(pars)
    
    return t, hp, hc


# Detector call function
def get_detector(ifo):
    """
    Gets the detector for the response function
    """
    if ifo not in ["H1","L1","V1"]:
        raise NameError("Detector '{0:s}' not recognized!".format(ifo))
    det = Detector(ifo)
    return det

# PSD call function
def get_psd(psd, psd_path, ifo=None, **kwargs):
    """
    Gets the PSD from either observing scenarios table or lalsimulation
    """
    # read f_low, f_high, and df or assume default values if not in kwargs
    f_low = kwargs["f_low"] if "f_low" in kwargs else 10.
    f_high = kwargs["f_high"] if "f_high" in kwargs else 2048.

    # try to read tables if strings are provided
    if type(psd)==str:
        if ifo in ["H1", "L1"]:
            psd_data = pd.read_csv(os.path.join(psd_path,'LIGO_P1200087.dat'), sep=' ', index_col=False)
        elif ifo in ["V1"]:
            psd_data = pd.read_csv(os.path.join(psd_path,'Virgo_P1200087.dat'), sep=' ', index_col=False)
        freqs = np.asarray(psd_data["freq"])
        # observeing scenarios table provides ASDs
        psd_vals = np.asarray(psd_data[psd])**2
        psd_interp = interp1d(freqs, psd_vals, bounds_error=False, fill_value=(np.inf,np.inf))

    # otherwise, assume lalsimulation psd was provided
    else:
        freqs = np.arange(f_low, f_high+df, df)
        psd_vals = np.asarray(list(map(psd, freqs)))
        psd_interp = interp1d(freqs, psd_vals, bounds_error=False, fill_value=(np.inf,np.inf))

    return psd_interp


# Sampling of extrinsic parameters
def sample_extrinsic():
    """
    Varies extrinsic parameters of waveform for calculating detection probability
    """

    # vary extrinsic parametrers
    ra, dec = np.random.uniform(0, 2*np.pi), np.arcsin(np.random.uniform(-1, 1))
    incl = np.arccos(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, np.pi)

    return ra, dec, incl, psi

# Calculate projection factor
def projection_factor(det, ra, dec, incl, psi):
    """
    Calculates projection factor Theta for a given detector antenna pattern and
    extrinsic parameters
    """

    # inclination
    gp, gc = (1 + np.cos(incl)**2)/2, np.cos(incl)

    # Use pycbc for detector response
    fp, fc = det.antenna_pattern(ra, dec, psi, 1e9)

    p = fp * gp
    c = fc * gc

    return np.sqrt(p**2 + c**2)


# SNR
def optimal_rho2(hf, Hf, freqs, psd, det, **kwargs):
    """
    Calculates:
    \rho^2 = 4 \int_{f_0}^{f_h} \frac{\tilde{h}^{\conj}(f)\tilde{h}(f)}{S(f)} df
    
    Takes in waveforms in the frequency domain
    
    hpf/hxf are the data, Hpf/Hxf are the templates
    """
    
    (hpf, hcf) = hf
    (Hpf, Hcf) = Hf
    
    fp, fc = 1., 0.
        
    template = (fp*Hpf + fc*Hcf)
    data = (fp*hpf + fc*hcf)
    
    df = freqs[1]-freqs[0]
    rho2 = 2 * np.sum( (np.conj(data)*template + np.conj(template)*data) * df  / psd(freqs) ).real

    return rho2





### Main detectability function ###

# Detection probability function
def detection_probability_eccentric(system, psd_path, ifos={"H1":"midhighlatelow"}, snr_thresh=8.0, Ntrials=1000, \
                                    aligned_spin=False, samp_rate=1024, **kwargs):
    """
    Calls other functions in this file to calculate a detection probability
    For multiprocessing purposes, takes in array 'system' of form:
    [m1, m2, z, (s1x,s1y,s1z), (s2x,s2y,s2z)]
    
    The function calculates 3 different detection probabilities: 
        1) assuming all systems are circular, and using circular templates
        2) using the proper system eccentricities and eccentric templates
        3) using the proper system eccentricities and circular templates
    """
    # get system parameters
    M1 = system[0]     # Msun
    M2 = system[1]     # Msun
    s1 = system[2]     # dimensionless spin magnitude
    s2 = system[3]     # dimensionless spin magnitude
    Deff = system[4]   # Mpc
    e0 = system[5]
    
    if aligned_spin==True:
        s1z = s1
        s2z = s2
    else:
        # get aligned spin magnituide from dimensionless spins assuming isotropy
        tilt1 = np.arccos(2*np.random.random() - 1)
        tilt2 = np.arccos(2*np.random.random() - 1)
        s1z = s1 * np.cos(tilt1)
        s2z = s2 * np.cos(tilt2)

    # read f_low, df or assume 10 Hz if not in kwargs
    f_low = kwargs["f_low"] if "f_low" in kwargs else 10.
    f_high = kwargs["f_high"] if "f_high" in kwargs else 2048.

    # get the detectors of choice for the response function
    detectors = {}
    for ifo in ifos.keys():
        detectors[ifo] = get_detector(ifo)

    # get the psds
    psds={}
    for ifo, psd in ifos.items():
        psd_interp = get_psd(psd, psd_path, ifo, f_low=f_low, f_high=f_high)
        psds[ifo] = psd_interp
    
    # arrays for optimal SNRs and detection probabilities
    snr_opt, pdet = np.zeros(3), np.zeros(3)

    # generate waveforms using both circular and eccentric templates
    t, hp, hc = gen_eccentric_waveform(M1, M2, s1z, s2z, Deff, 0.0, modes=[[2,2]], f0=f_low, srate=samp_rate)
    dt = t[1]-t[0]
    hpf, hcf = JBJF(hp, hc, dt)
    
    t_ecc, hp_ecc, hc_ecc = gen_eccentric_waveform(M1, M2, s1z, s2z, Deff, e0, modes=[[2,2]], f0=f_low, srate=samp_rate)
    dt_ecc = t_ecc[1]-t_ecc[0]
    hpf_ecc, hcf_ecc = JBJF(hp_ecc, hc_ecc, dt_ecc)
    
    assert dt==dt_ecc, "Sampling rate is different between the circular and eccentric waveforms!"
    
    
    ### First, check that the optimal matched-filter SNR is greater than the SNR threshold for either waveform
    freqs = np.fft.rfftfreq(len(hp),d=dt)
    snr_opts={}
    for ifo in ifos.keys():
        snr_opts[ifo] = np.sqrt(optimal_rho2((hpf, hcf), (hpf, hcf), freqs, psds[ifo], detectors[ifo]))
    snr_opt_circ =  np.linalg.norm(list(snr_opts.values()))
    
    freqs = np.fft.rfftfreq(len(hp_ecc),d=dt_ecc)
    snr_opts={}
    for ifo in ifos.keys():
        snr_opts[ifo] = np.sqrt(optimal_rho2((hpf_ecc, hcf_ecc), (hpf_ecc, hcf_ecc), freqs, psds[ifo], detectors[ifo]))
    snr_opt_ecc =  np.linalg.norm(list(snr_opts.values()))
    
    if (snr_opt_circ < snr_thresh) and (snr_opt_ecc < snr_thresh):
        # don't bother calculating the eccentric waveform SNR with circular template
        snr_opt[0] = snr_opt_circ
        snr_opt[1] = snr_opt_ecc
        snr_opt[2] = np.nan
        return snr_opt, pdet
    
    
    ### Timeslide to see where the SNR is maximized (this will take time...)
    max_wf_length = np.maximum(len(hp), len(hp_ecc))
    freqs = np.fft.rfftfreq(max_wf_length,d=dt)
    timeslide_rho2 = []
    if len(hp) < max_wf_length:
        hpf_ecc, hcf_ecc = JBJF(hp_ecc, hc_ecc, dt_ecc)
        zero_pad = np.zeros(max_wf_length - len(hp))
        for idx in np.arange(len(zero_pad)):
            start_pad, end_pad = zero_pad[:idx], zero_pad[idx:]
            hp_tmp = np.append(np.append(start_pad, hp), end_pad)
            hc_tmp = np.append(np.append(start_pad, hc), end_pad)
            hpf_tmp, hcf_tmp = JBJF(hp_tmp, hc_tmp, dt)
            timeslide_rho2.append(optimal_rho2((hpf_tmp, hcf_tmp), (hpf_ecc, hcf_ecc), freqs, \
                                                psds[list(ifos.keys())[0]], detectors[list(ifos.keys())[0]]))
        max_idx = np.argmax(timeslide_rho2)
        start_pad, end_pad = zero_pad[:max_idx], zero_pad[max_idx:]
        hp = np.append(np.append(start_pad, hp), end_pad)
        hc = np.append(np.append(start_pad, hc), end_pad)
        hpf, hcf = JBJF(hp, hc, dt)
    elif len(hp_ecc) < max_wf_length:
        hpf, hcf = JBJF(hp, hc, dt)
        zero_pad = np.zeros(max_wf_length - len(hp_ecc))
        for idx in np.arange(len(zero_pad)):
            start_pad, end_pad = zero_pad[:idx], zero_pad[idx:]
            hp_tmp = np.append(np.append(start_pad, hp_ecc), end_pad)
            hc_tmp = np.append(np.append(start_pad, hc_ecc), end_pad)
            hpf_tmp, hcf_tmp = JBJF(hp_tmp, hc_tmp, dt)
            timeslide_rho2.append(optimal_rho2((hpf, hcf), (hpf_tmp, hcf_tmp), freqs, \
                                                psds[list(ifos.keys())[0]], detectors[list(ifos.keys())[0]]))
        max_idx = np.argmax(timeslide_rho2)
        start_pad, end_pad = zero_pad[:max_idx], zero_pad[max_idx:]
        hp_ecc = np.append(np.append(start_pad, hp_ecc), end_pad)
        hc_ecc = np.append(np.append(start_pad, hc_ecc), end_pad)
        hpf_ecc, hcf_ecc = JBJF(hp_ecc, hc_ecc, dt_ecc)
    else:
        # same waveform for template and signal
        hpf, hcf = JBJF(hp, hc, dt)
        hpf_ecc, hcf_ecc = JBJF(hp_ecc, hc_ecc, dt_ecc)
    
    
    circular = (hp, hc, hpf, hcf)
    eccentric = (hp_ecc, hc_ecc, hpf_ecc, hcf_ecc)
    
    ### Loop through the three methods ###
    for idx, (template, signal) in enumerate(zip([circular, eccentric, circular], [circular, eccentric, eccentric])): 
        
        # unpack tuples with signal and template
        (hp, hc, hpf, hcf) = signal
        (Hp, Hc, Hpf, Hcf) = template

        snr_opts={}
        for ifo in ifos.keys():
            snr_opts[ifo] = np.sqrt(optimal_rho2((hpf, hcf), (Hpf, Hcf), freqs, psds[ifo], detectors[ifo]))
        snr_opt[idx] =  np.linalg.norm(list(snr_opts.values()))

        # if the combined SNR is less than the detection threshold, give weight of 0 (this is redundant with above)
        if snr_opt[idx] < float(snr_thresh):
            pdet[idx] = 0.0

        # otherwise, calculate the matched-filter SNR 
        else:
            snrs = []
            for n in range(Ntrials):
                network_snr = []
                ra, dec, incl, psi = sample_extrinsic()
                for ifo, det in detectors.items():
                    rho_factor = projection_factor(det, ra, dec, incl, psi)
                    network_snr.append(snr_opts[ifo]*rho_factor)
                snrs.append(np.linalg.norm(network_snr))
            # now, we see what percentage of SNRs passed our threshold
            snrs = np.asarray(snrs)
            passed = sum(1 for i in snrs if i>=float(snr_thresh))
            pdet[idx] = float(passed) / len(snrs)
        
    # return all 3 optimal SNRs and detection probabilities
    return snr_opt, pdet

