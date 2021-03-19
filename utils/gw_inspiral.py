import numpy as np
from scipy import integrate
from scipy.integrate import ode
from scipy.optimize import brentq


"""
Functions for integrating binary inspirals and calculating eccentricities
"""


def calc_c0(a0,e0):
    num = a0*(1-e0**2)
    denom = e0**(12./19) * (1+(121./304)*e0**2)**(870./2299)
    return num/denom

def inspiral_time_peters(a0,e0,m1,m2,af=0):
    """
    Computes the inspiral time, in Gyr, for a binary
    a0 in Au, and masses in solar masses

    if different af is given, computes the time from a0,e0
    to that final semi-major axis

    for af=0, just returns inspiral time
    for af!=0, returns (t_insp,af,ef)
    """
    coef = 6.086768e-11 #G^3 / c^5 in au, gigayear, solar mass units
    beta = (64./5.) * coef * m1 * m2 * (m1+m2)

    if e0 == 0:
        if not af == 0:
            print("ERROR: doesn't work for circular binaries")
            return 0
        return a0**4 / (4*beta)

    c0 = a0 * (1.-e0**2.) * e0**(-12./19.) * (1.+(121./304.)*e0**2.)**(-870./2299.)

    if af == 0:
        eFinal = 0.
    else:
        r = ode(deda_peters)
        r.set_integrator('lsoda')
        r.set_initial_value(e0,a0)
        r.integrate(af)
        if not r.successful():
            print("ERROR, Integrator failed!")
        else:
            eFinal = r.y[0]

    time_integrand = lambda e: e**(29./19.)*(1.+(121./304.)*e**2.)**(1181./2299.) / (1.-e**2.)**1.5
    integral,abserr = integrate.quad(time_integrand,eFinal,e0)

    if af==0:
        return integral * (12./19.) * c0**4. / beta
    else:
        return (integral * (12./19.) * c0**4. / beta,af,eFinal)

def a_at_fLow(m1,m2,fLow=10):
    """
    Computes the semi-major axis at a *GW* frequency of fLow
    Masses in solar masses, fLow in Hz
    """
    G = 3.9652611e-14 # G in au,solar mass, seconds
    quant = G*(m1+m2) / (4*np.pi**2 * (fLow/2)**2)
    return quant**(1./3)

def Rperi_at_eccentricity(a0,e0,e):
    """
    Computes periapse distance at a specified eccentricity, given
    starting orbital conditions a0 and e0

    a in AU
    """
    c0 = calc_c0(a0,e0)

    num = c0*e**(12./19) * (1 + (121./304)*e**2)**(870./2299) * (1-e)
    denom = (1-e**2)
    return num/denom

def Rapo_at_eccentricity(a0,e0,e):
    """
    Computes apoapse distance at a specified eccentricity, given
    starting orbital conditions a0 and e0

    a in AU
    """
    c0 = calc_c0(a0,e0)

    num = c0*e**(12./19) * (1 + (121./304)*e**2)**(870./2299) * (1+e)
    denom = (1-e**2)
    return num/denom

def a_at_eccentricity(a0,e0,e):
    """
    Computes semi-major axis at a specified eccentricity, given
    starting orbital conditions a0 and e0

    a in AU
    """
    c0 = calc_c0(a0,e0)

    num = c0*e**(12./19) * (1 + (121./304)*e**2)**(870./2299)
    denom = (1-e**2)
    return num/denom

def eccentricity_at_a(m1,m2,a_0,e_0,a):
    """
    Computes the eccentricity at a given semi-major axis a

    Masses are in solar masses, a_0 in AU

    """
    r = ode(deda_peters)
    r.set_integrator('lsoda')
    r.set_initial_value(e_0,a_0)

    r.integrate(a)

    if not r.successful():
        print("ERROR, Integrator failed!")
    else:
        return r.y[0]

def eccentric_gwave_freq(a,m,e):
    """
    returns the gravitational-wave frequency for a binary at seperation a (AU), mass
    M (solar masses), and eccentricity e, using the formula from Wen 2003
    """

    freq = 1 / (86400*au_to_period(a,m))
    return 2*freq*pow(1+e,1.1954)/pow(1-e*e,1.5)

def eccentricity_at_eccentric_fLow(m1,m2,a_0,e_0,fLow=10,retHigh = False):
    """
    Computes the eccentricity at a given fLow using the peak frequency from Wen
    2003

    Masses are in solar masses, a_0 in AU.  The frequency here is the
    gravitational-wave frequency.

    Note that it is possible that, for binaries that merge in fewbody, there is
    no solution, since the binary was pushed into the LIGO band above 10Hz.  In
    this case, there is no a/e from that reference that will give you 10Hz, so
    this just returns e_0 by default, and 0.99 if retHigh is true
    """

    ecc_at_a = lambda a: eccentricity_at_a(m1,m2,a_0,e_0,a)
    freq_at_a = lambda a: eccentric_gwave_freq(a,m1+m2,ecc_at_a(a))
    zero_eq = lambda a: freq_at_a(a) - fLow

    lower_start = zero_eq(1e-10)
    upper_start = zero_eq(1)

    if (np.sign(lower_start) == np.sign(upper_start) or
        np.isnan(lower_start) or np.isnan(upper_start)):
        if retHigh:
            return 0.99
        else:
            return e_0
    else:
        a_low = brentq(zero_eq,1e-10,1)
        return ecc_at_a(a_low)

def eccentricity_at_ref_freq(m1, m2, a0, e0, f_gw=10, e_ref=0.999, R_peri=False):

    """
    Main function for calculating eccentricities at a given gravitational-wave frequency
    """

    # calculate c0
    c0 = peters.calc_c0(a0, e0)
    # calculate what the periapse distance was at reference high eccentricity
    r_peri_high_e = peters.Rperi_at_eccentricity(a0,e0,e_ref)
    # calculate eccentricity and inspiral times in various ways
    a_low = peters.a_at_fLow(m1, m2, fLow=f_gw/2)

    if R_peri==False:
        ### METHOD FOR ECCENTRICITY CALCULATION: Wen 2003 ###
        ef = peters.eccentricity_at_eccentric_fLow(m1,m2,a0,e0, fLow=f_gw, retHigh=True)

    else:
        ### METHOD FOR ECCENTRICITY CALCULATION: USING Rperi ###
        # if r_peri at high e is less than a_low, then the binary formed in band
        if r_peri_high_e <= a_low:
            ef = peters.eccentricity_at_eccentric_fLow(m1, m2, a_low, e_ref, fLow=f_gw, retHigh=False)
        else:
            # bianry evolved into the LIGO band
            ef = peters.eccentricity_at_eccentric_fLow(m1, m2, a0, e0, fLow=f_gw, retHigh=False)
            
    return ef
