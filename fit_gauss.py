import math
import numpy as np
import sys
from numpy.random import *

import scipy.optimize as so
import h5py
import argparse
import misc_data_io as misc

n_freq = 1024
n_corr = 36

def gaussian(p,x):
    """
    p[0] - Constant offset
    p[1] - Slope
    p[2] - Amplitude
    p[3] - Sigma
    p[4] - Mean
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

def beam_fit(Data, RA, dec=0):
    """ Gets a beam fits in frequency.
    returns:
    """

    autos = [  0.,   8.,  15.,  21.,  26.,  30.,  33.,  35.]
    beam_fit = np.zeros([Data.shape[0], Data.shape[1], 5])

    print "Number of frequencies:", n_freq
    
    for freq_ind in range(n_freq):
        for auto_corr in autos:
            p = run_fit(Data, freq_ind, RA, correlation=auto_corr)[0]
            beam_fit[freq_ind, auto_corr, :] = p

    beam_fit[3] = 2.35 * np.cos(np.deg2rad(dec)) * beam_fit[3]

    return beam_fit

# Create a dictionary with each fitting object's information in the form: {"Obj": [RA_min, RA_max, Declination]}
celestial_object = { "CasA": [344, 358, 58.83] }#, "TauA": [77, 87, 83.6], "CygA": [297, 302, 40.73]}

parser = argparse.ArgumentParser(description="This programs tries to fit beam from point-source transits.")
parser.add_argument("Data", help="Directory containing acquisition files.")
args = parser.parse_args()

#parser.add_argument("Object", help="")
#parser.add_argument("", help="")
#parser.add_argument("", help="")

files = np.str(args.Data) + '/*h5*'
files = args.Data
print "Reading in Data"

Data, vis, utime, RA = misc.get_data(files)

for src in celestial_object.keys():
    print "Fitting", src
    
    vis_obj = vis[:, :, ( RA > celestial_object[src][0] ) & ( RA < celestial_object[src][1])]
    RA_src = RA[ ( RA > celestial_object[src][0] ) & ( RA < celestial_object[src][1]) ]
    
    beam_params = beam_fit(vis_obj, RA_src, dec = celestial_object[src][2])

    t = misc.eph.datetime.fromtimestamp(utime[0])
    date_str = t.strftime('%Y_%m_%d_%H:%M')
    filename = '/scratch/k/krs/connor/beam_fit' + src + date_str + '.hdf5'

    g = h5py.File(filename,'w')
    g.create_dataset('beam', data = beam_params)
    g.create_dataset('visibilities', data = vis_obj)
    g.create_dataset('RA', data = RA_src)
    g.close()

    print "Saved data in", filename
