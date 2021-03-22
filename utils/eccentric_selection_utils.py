import numpy as np
import pandas as pd
import os

from scipy.interpolate import interp1d
from scipy import signal
from scipy.optimize import minimize

from pycbc.detector import Detector

import EOBRun_module




#####################################################################
### FUNCTIONS USED FOR SNR AND DETECTION PROBABILITY CALCULATIONS ###
#####################################################################



### WAVEFORM GENERATION ###
def modes_to_k(modes):
    """
    Map (l,m) -> k
    """
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

def fourier_transform(h,dt):
    """
    Fourier transform of TD waveform
    """
    (hp, hc) = h
    
    hpf = np.fft.rfft(hp) * dt
    hcf = np.fft.rfft(-hc) * dt
    return hpf, hcf

def inverse_fourier_transform(hf,n,dt):
    """
    Inverse Fourier transform of FD waveform
    
    N is the length of the length of the transformed axis that is output
    """
    (hpf, hcf) = hf
    
    hp = np.fft.irfft(hpf, n) / dt
    hc = np.fft.irfft(-hcf, n) / dt
    return hp, hc

def gen_eccentric_waveform(M1, M2, Chi1, Chi2, Deff, e0, Iota=np.pi/2, modes=[[2,2]], phase=0.0, f0=10, srate=4096):
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
        'interp_uniform_grid'      : 1,
        'df'                       : 0.01,    #df for FD interpolation
        'coalescence_angle'        : phase
    }
    
    t, hp, hc = EOBRun_module.EOBRunPy(pars)
    
    return t, (hp, hc)



### DETECTORS AND PSDS ##

def get_detector(ifo):
    """
    Gets the detector for the response function
    """
    if ifo not in ["H1","L1","V1"]:
        raise NameError("Detector '{0:s}' not recognized!".format(ifo))
    det = Detector(ifo)
    return det

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



### EXTRINSIC PARAMETERS ###

def sample_extrinsic():
    """
    Varies extrinsic parameters of waveform for calculating detection probability
    """

    # vary extrinsic parametrers
    ra, dec = np.random.uniform(0, 2*np.pi), np.arcsin(np.random.uniform(-1, 1))
    incl = np.arccos(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, np.pi)

    return ra, dec, incl, psi

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



### SNR-RELATED FUNCTIONS ###

def noise_weighted_inner_product(a, b, freqs, psd_interp):
    """
    Noise weighted inner product
    
    Frequency-domain signal `b` gets the complex conjugate
    """
    (ap, ac) = a
    (bp, bc) = b
    df = freqs[1]-freqs[0]
    
    inner = 4 * df * np.sum((ap * np.conj(bp) + ac * np.conj(bc)) / psd_interp(freqs))
    
    return inner


def overlap(sf, hf, freqs, psd_interp):
    """
    Calculates overlap between two waveforms, a = (spf,scf) and b = (hpf, hcf):
    O = inner(sf, hf) = / SQRT(inner(sf, sf) + inner(hf, hf))
    where `inner` is the noise-weighted inner product
    """
   
    inner_ss = noise_weighted_inner_product(sf, sf, freqs, psd_interp)
    inner_hh = noise_weighted_inner_product(hf, hf, freqs, psd_interp)
    inner_sh = noise_weighted_inner_product(sf, hf, freqs, psd_interp)
    
    overlap = inner_sh / np.sqrt(inner_ss * inner_hh)
    return overlap.real

def matched_filter_snr(sf, hf, freqs, psd_interp, **kwargs):
    """
    Calculates matched-filter SNR
    
    sf is the signal, hf is the template
    
    Takes in waveforms in the frequency domain
    """
    
    (spf, scf) = sf
    (hpf, hcf) = hf

    df = freqs[1]-freqs[0]
    rho_mf = noise_weighted_inner_product(sf, hf, freqs, psd_interp) / optimal_snr_squared(hf, freqs, psd_interp)**0.5
    
    return rho_mf

def optimal_snr_squared(hf, freqs, psd_interp, **kwargs):
    """
    Calculates:
    \rho^2 = 4 \int_{f_0}^{f_h} \frac{\tilde{h}^{\conj}(f)\tilde{h}(f)}{S(f)} df
    
    Takes in waveforms in the frequency domain
    
    Returns rho2 as a complex number
    """
    
    df = freqs[1]-freqs[0]
    rho2 = noise_weighted_inner_product(hf, hf, freqs, psd_interp)
    
    return rho2

def chi_squared(sf, hf, freqs, psd_interp, p=10000, f_low=10, f_high=2048):
    """
    Calculates chi^2 statistic as in Eq. 5 of GW150914 paper
    
    `p` specifies the frequency resolution at which to bin sub-templates
    """
    (spf, scf) = sf
    (hpf, hcf) = hf
    
    p = len(freqs)
    df = freqs[1]-freqs[0]
    freq_bins = np.linspace(f_low, f_high, p+1)
    
    inner_hh = noise_weighted_inner_product(hf, hf, freqs, psd_interp)
    inner_sh = noise_weighted_inner_product(sf, hf, freqs, psd_interp)
    
    """bin_by_bin_overlap = []
    for fl, fh in zip(freq_bins[:-1], freq_bins[1:]):
        idxs_in_fbin = np.where((freqs > fl) & (freqs <= fh))
        bin_by_bin_overlap.append(np.sum(4 * df * (spf[idxs_in_fbin] * np.conj(hpf[idxs_in_fbin]) + \
                                            scf[idxs_in_fbin] * np.conj(hcf[idxs_in_fbin])) / psd_interp(freqs[idxs_in_fbin])))
    bin_by_bin_overlap = np.asarray(bin_by_bin_overlap)"""
    bin_by_bin_overlap = 4 * df * (spf * np.conj(hpf) + scf * np.conj(hcf)) / psd_interp(freqs)
    
    chi2 = p/(2*p-2) * inner_hh**-1 * np.sum(np.abs(bin_by_bin_overlap - inner_sh/p)**2)

    return chi2

def adjusted_snr(rho, chi2):
    """
    Calculates SNR adjusted by chi^2 as in Eq. 6 of GW150914 paper
    
    `df` is the frequency resolution at which to bin sub-templates
    """
    if chi2 <= 1:
        return rho
    else:
        return rho / ((1 + chi2**3)/2)**(1./6)
    
    
    
### WAVEFORM PROCESSING ###

def zero_pad_and_align_maxima(s, h, ts, th):
    """
    Zero-pad the shorter of two time-domain waveforms
    
    `s` and `h` should be tuples with (+,x) polarizations
    """
    (sp, sc) = s
    (hp, hc) = h
    max_wf_length = np.max([len(sp),len(hp)])
    max_time_s = ts[np.argmax(np.abs(sp) + np.abs(sc))]
    max_time_h = th[np.argmax(np.abs(hp) + np.abs(hc))]
    timeshift = np.abs(max_time_s - max_time_h)
    
    if len(sp) < max_wf_length:
        Npad = max_wf_length - len(sp)
        sp = np.concatenate((np.zeros(Npad), sp))
        sc = np.concatenate((np.zeros(Npad), sc))
        
        max_idx_h = np.argmax(np.abs(hp) + np.abs(hc))
        max_idx_s = np.argmax(np.abs(sp) + np.abs(sc))
        sp = np.roll(sp, max_idx_h-max_idx_s)
        sc = np.roll(sc, max_idx_h-max_idx_s)
        
        dts = ts[1]-ts[0]
        added_times = ts[-1] + [(dts*(i+1)) for i in np.arange(int(Npad))]
        ts = np.concatenate((ts, np.asarray(added_times)))
        
    elif len(hp) < max_wf_length:
        Npad = max_wf_length - len(hp)
        hp = np.concatenate((np.zeros(Npad), hp))
        hc = np.concatenate((np.zeros(Npad), hc))
        
        max_idx_h = np.argmax(np.abs(hp) + np.abs(hc))
        max_idx_s = np.argmax(np.abs(sp) + np.abs(sc))
        hp = np.roll(hp, max_idx_s-max_idx_h)
        hc = np.roll(hc, max_idx_s-max_idx_h)
        
        dth = th[1]-th[0]
        added_times = th[-1] + [(dth*(i+1)) for i in np.arange(int(Npad))]
        th = np.concatenate((th, np.asarray(added_times)))
        
    return (sp,sc), (hp,hc), ts, th, timeshift
    
def window_waveform(t, h, method='tukey', alpha=0.05):
    """
    Apply window to time-domain waveform `h`, according to `method`
    """
    if method not in ['tukey']:
        raise ValueError("The windowing function you specified ({}) is not defined!".format(metohd))
        
    (hp, hc) = h
    wf_length = len(hp)
    dt = t[1]-t[0]
    if method=='tukey':
        # first, pad the upper end with zeros so the merger isn't swallowed up
        hp = np.concatenate((hp, np.zeros(int(alpha*wf_length))))
        hc = np.concatenate((hc, np.zeros(int(alpha*wf_length))))
        added_times = t[-1] + [(dt*(i+1)) for i in np.arange(int(alpha*wf_length))]
        t = np.concatenate((t, np.asarray(added_times)))
        wf_length = len(hp)
        
        # apply window
        hp = np.multiply(signal.tukey(M=wf_length, alpha=alpha), hp)
        hc = np.multiply(signal.tukey(M=wf_length, alpha=alpha), hc)
        
        return t, (hp,hc)
        
    
        
### MAXIMIZING OVER PHASE AND TIME ###    

def apply_timeshift(hf, freqs, timeShift):
    """
    Applies timeshift to waveform `hf` in frequency domain
    
    `hf` should be a tuple with (+,x) polarizations
    """
    
    (hpf, hcf) = hf
    duration = 1.0/(freqs[1]-freqs[0])
    
    ### FIXME: Should the duration added to the timeshift here???
    #hpf_shift = hpf * np.exp(-2j * np.pi * (duration + timeShift) * freqs)
    #hcf_shift = hcf * np.exp(-2j * np.pi * (duration + timeShift) * freqs)
    hpf_shift = hpf * np.exp(-2j * np.pi * timeShift * freqs)
    hcf_shift = hcf * np.exp(-2j * np.pi * timeShift * freqs)
    
    return (hpf_shift, hcf_shift)

def apply_phaseshift(hf, phaseShift):
    """
    Applies phaseshhift to waveform `hf` in frequency domain
    
    `hf` should be a tuple with (+,x) polarizations
    """
    
    (hpf, hcf) = hf
    
    hpf_shift = hpf * np.exp(-2j * phaseShift)
    hcf_shift = hcf * np.exp(-2j * phaseShift)
    
    return (hpf_shift, hcf_shift)

def maximize_overlap_optimizable(time_phase, *args):
    """
    Function that can be minimized to find the optimal phase and time
    
    Takes in [time_shift, phase_shift] and additional arguments
    
    Returns -1*overlap
    """
    timeShift = time_phase[0]
    phaseShift = time_phase[1]
    
    sf, hf, freqs, psd_interp = (args)
    
    # apply phase and time shift to template
    hf_shift = apply_phaseshift(hf, phaseShift)
    hf_shift = apply_timeshift(hf_shift, freqs, timeShift)
    
    return -overlap(sf, hf_shift, freqs, psd_interp)

def maximize_time_phase(sf, hf, freqs, psd_interp, time_limit_dt=1e-3, return_time_phase=False):
    """
    Finds the time and phase shift that maximize the overlap between the waveforms
    
    Returns adjusted template `hf` in the Fourier domain
    
    `time_limit_dt` sets the fraction of the total duration by which to vary times from the joint maxima
    
    If `return_time_phase` is set to True, also returns the time and phase shifts
    """
    
    # set limits and initial guesses
    duration = 1.0 / (freqs[1] - freqs[0])
    time_limit = time_limit_dt * duration
    phase_limit = np.pi / 2
    timeShift_guess = 0
    phaseShift_guess = 0
    
    x0 = [timeShift_guess, phaseShift_guess]
    bounds = [(timeShift_guess-time_limit, timeShift_guess+time_limit),\
              (phaseShift_guess-phase_limit, phaseShift_guess+phase_limit)]
    args = (sf, hf, freqs, psd_interp)
    
    # minimize overlap
    res = minimize(maximize_overlap_optimizable, bounds=bounds, x0=x0, args=args, tol=1e-10, method='Powell')
    timeShift, phaseShift = res.x[0], res.x[1]
    maximum_overlap = -res.fun
    
    # shift template according to this timeshift and phaseshift
    hf_shift = apply_phaseshift(hf, phaseShift)
    hf_shift = apply_timeshift(hf_shift, freqs, timeShift)
    
    
    if return_time_phase==True:
        return hf_shift, timeShift, phaseShift
    else:
        return hf_shift


### MAIN DETECTABILITY FUNCTION ###
def detection_probability_eccentric(system, psd_path, ifos={"H1":"midhighlatelow"}, snr_thresh=8.0, Ntrials=1000, aligned_spin=False, **kwargs):
    """
    Calls other functions in this file to calculate a detection probability
    For multiprocessing purposes, takes in array 'system' of form:
    [m1, m2, s1, s2, dL, e(fLow)]
    
    The function calculates 3 different detection probabilities: 
        1) assuming all systems are circular, and using circular templates
        2) using the proper system eccentricities and eccentric templates
        3) using the proper system eccentricities and circular templates
        
    Throughout this, `s` is the signal, `h` is the template
    """
    # get system parameters
    M1 = system[0]     # Msun
    M2 = system[1]     # Msun
    s1 = system[2]     # dimensionless spin magnitude
    s2 = system[3]     # dimensionless spin magnitude
    Deff = system[4]   # Mpc
    e0 = system[5]
    
    # first, get aligned spin magnituide from dimensionless spins assuming isotropy
    if aligned_spin == True:
        s1z = s1
        s2z = s2
    else:
        tilt1 = np.arccos(2*np.random.random() - 1)
        tilt2 = np.arccos(2*np.random.random() - 1)
        s1z = s1 * np.cos(tilt1)
        s2z = s2 * np.cos(tilt2)

    # read f_low/f_high/srate or assume 10Hz/2048Hz/1024Hz if not in kwargs
    f_low = kwargs["f_low"] if "f_low" in kwargs else 10.
    f_high = kwargs["f_high"] if "f_high" in kwargs else 2048.
    srate = kwargs["srate"] if "srate" in kwargs else 4096.

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
    snr_opt, chi2, pdet = np.zeros(3), np.zeros(3), np.zeros(3)

    # generate waveforms using both circular and eccentric templates
    th, h = gen_eccentric_waveform(M1, M2, s1z, s2z, Deff, 0.0, modes=[[2,2]], srate=srate, f0=f_low)
    ts, s = gen_eccentric_waveform(M1, M2, s1z, s2z, Deff, e0, modes=[[2,2]], srate=srate, f0=f_low)
    dt = th[1]-th[0]
    assert dt==ts[1]-ts[0], "Sampling rate is different between the circular and eccentric waveforms!"
    
    # window waveforms
    th, h = window_waveform(th, h)
    ts, s = window_waveform(ts, s)
    
    # zero-pad and align waveforms
    s, h, ts, th, shift = zero_pad_and_align_maxima(s, h, ts, th)
    assert len(s[0])==len(h[0]), "Waveforms are still different lengths after zero-padding!"
    
    # fourier transform
    hf = fourier_transform(h, dt)
    sf = fourier_transform(s, dt)
    max_wf_length = max(len(h[0]),len(s[0]))   # should be the same, since we already zero-padded
    freqs = np.fft.rfftfreq(max_wf_length,d=dt)


    # First, check that the optimal matched-filter SNR greater than the SNR threshold for both waveforms
    snr_opts={}
    chi2_snr={}
    for idx, ifo in enumerate(ifos.keys()):
        rho2 = optimal_snr_squared(hf, freqs, psds[ifo])
        snr_opts[ifo] = np.sqrt(rho2.real)
        if idx==0:
            chi2_hh = chi_squared(hf, hf, freqs, psds[ifo]).real
    snr_opt_hh =  np.linalg.norm(list(snr_opts.values()))
    
    snr_opts={}
    chi2_snr={}
    for ifo in ifos.keys():
        rho2 = optimal_snr_squared(sf, freqs, psds[ifo])
        snr_opts[ifo] = np.sqrt(rho2).real
        if idx==0:
            chi2_ss = chi_squared(sf, sf, freqs, psds[ifo]).real
    snr_opt_ss =  np.linalg.norm(list(snr_opts.values()))
    
    if (snr_opt_hh < snr_thresh) and (snr_opt_ss < snr_thresh):
        # don't bother calculating the eccentric waveform SNR with circular template
        snr_opt[0] = snr_opt_hh
        snr_opt[1] = snr_opt_ss
        snr_opt[2] = np.nan
        chi2[0] = chi2_hh
        chi2[1] = chi2_ss
        chi2[2] = np.nan
        return snr_opt, chi2, pdet
    
    
    # Maximize over time and phase
    hf = maximize_time_phase(sf, hf, freqs, psd_interp, time_limit_dt=1e-3, return_time_phase=False)
    
    ### Loop through the three methods ###
    signals = [hf, sf, sf]
    templates = [hf, sf, hf]
    for idx, (signal, template) in enumerate(zip(signals, templates)): 

        # get maximum SNR with optimal extrinsic parameters
        snr_max_mf={}
        chi2_snr_max_mf={}
        for iidx, ifo in enumerate(ifos.keys()):
            snr_mf = matched_filter_snr(signal, template, freqs, psds[ifo])
            snr_max_mf[ifo] = snr_mf.real
            if iidx==0:
                chi2_mf = chi_squared(signal, template, freqs, psds[ifo])
        snr_opt[idx] =  np.linalg.norm(list(snr_max_mf.values()))
        chi2[idx] = chi2_mf.real

        # if the combined SNR is less than the detection threshold, give weight of 0 (this is redundant with above for the optimal SNRs)
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
                    network_snr.append(snr_opt[idx]*rho_factor)
                snrs.append(np.linalg.norm(network_snr))
            # now, we see what percentage of SNRs passed our threshold
            snrs = np.asarray(snrs)
            passed = sum(1 for i in snrs if i>=float(snr_thresh))
            pdet[idx] = float(passed) / len(snrs)
        
    # return all 3 optimal SNRs and detection probabilities
    return snr_opt, chi2, pdet
