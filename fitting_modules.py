import math
import numpy as np
import sys
from numpy.random import *

import scipy.optimize as so
import ch_util.ephemeris as eph

n_freq = 1024
n_corr = 36

def gaussian(p,x):
    """ Generates a gaussian with a linear offset
    
    Parameters
    ----------
    p : list
      gaussian params  
    
      p[0] - Constant offset
      p[1] - Slope
      p[2] - Amplitude
      p[3] - Sigma
      p[4] - Mean
    x : array_like
      domain of gaussian
    
    Returns
    -------
    gauss : array_like
      gaussian function offset by a line
    """
    gauss = p[0] + p[1]*x + p[2] * (p[3]**2 * 2 * np.pi)**(-0.5) * np.exp(-(x - p[4])**2 / 2. / p[3]**2)

    return gauss

def diff(p, x, Data):
    diff = gaussian(p,x) - Data

    return abs(diff)

def run_fit(Arr, freq, RA, correlation=0):
    
    Data = abs(Arr[freq, correlation, :])
    x = RA

    A = Data.max() - Data.min()
    FWHM = 7
    sig = FWHM / 2.3548
    C = Data.min()
    x_c = np.median(x)
    m = (Data[-1] - Data[0]) / (RA[-1] - RA[0])
    
    param_guess = [C, m, A, sig, x_c]
    param_fit = so.leastsq(diff, param_guess, args=(x,Data), maxfev=10000)[0]

    return param_fit, Data, x

def beam_fit(Data, RA, dec=0, correlations="autos"):
    """ Gets a beam fits in frequency.

    Parameters
    ----------
    Data  : array_like
      Data array shaped as (n_freq, ncorr, ntimes)
    RA  : array_like
      Right ascension, length ntimes
    dec : float
      Declination of object being fit. Assumes zero.
    correlations  : string
      Which correlations to fit
    
    Returns
    -------
    beam_fit  : list
      best fit parameters of gaussian ordered by [C, m, A, sig, x_c] 
      where offset line is y=mx+C, A is gaussian amplitude, sig is stdev, and x_c is centroid
    """

    if correlations=='autos':
        correlations = [  0.,   8.,  15.,  21.,  26.,  30.,  33.,  35.]
    elif correlations=='All':
        correlations = np.arange(Data.shape[1])
    else:
        print "Error: Need proper set of correlations"
    
    beam_fit = np.zeros([Data.shape[0], Data.shape[1], 5])

    print "Number of frequencies:", n_freq
    
    for freq_ind in range(n_freq):
        if (freq_ind % 256)==0: print "Done fitting freq:", freq_ind
        for corr in correlations:
            p = run_fit(Data, freq_ind, RA, correlation=corr)[0]
            beam_fit[freq_ind, corr, :] = p

    beam_fit[3] = 2.35 * np.cos(np.deg2rad(dec)) * beam_fit[3]

    return beam_fit
