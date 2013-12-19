import math
import numpy as np
import sys
from numpy.random import *

import scipy.optimize as so
import h5py
import argparse
import misc_data_io as misc
import fitting_modules as fm

n_freq = 1024
n_corr = 36

parser = argparse.ArgumentParser(description="This programs tries to fit beam from point-source transits.")
parser.add_argument("Data", help="Directory containing acquisition files.")
parser.add_argument("--Objects", help="Celestial objects to fit", default='All')
args = parser.parse_args()

files = np.str(args.Data) + '/*h5*'
print "Reading in Data"

Data, vis, utime, RA = misc.get_data(files)

RA_sun = eph.transit_RA(eph.solar_transit(utime[0]))
sun_RA_low = RA_sun - 6
sun_RA_low = RA_sun + 6

# Create a dictionary with each fitting object's information in the form: {"Obj": [RA_min, RA_max, Declination]}
celestial_object = { "CasA": [344, 358, 58.83], "TauA": [77, 87, 83.6], "CygA": [297, 302, 40.73]}

parser = argparse.ArgumentParser(description="This programs tries to fit beam from point-source trans\
its.")
parser.add_argument("Data", help="Directory containing acquisition files.")
parser.add_argument("--Objects", help="Celestial objects to fit", default='All')
parser.add_argument("--maxfile", help="Maxfile number e.g. 0051", default="")
args = parser.parse_args()

files = np.str(args.Data) + '*h5' + args.maxfile + '*'
print files
print "Reading in Data"

Data, vis, utime, RA = misc.get_data(files)

if args.Objects != "All":
    srcs2fit = [args.Objects]
else:
    srcs2fit = celestial_object.keys()

for src in srcs2fit:

    print "Fitting", src

    vis_obj = vis[:, :, ( RA > celestial_object[src][0] ) & ( RA < celestial_object[src][1])]
    RA_src = RA[ ( RA > celestial_object[src][0] ) & ( RA < celestial_object[src][1]) ]

    if len(RA_src) == 0:
        pass
    else:
        print "Shape:", RA_src.shape

        beam_params = fm.beam_fit(vis_obj, RA_src, dec = celestial_object[src][2])

        transit_time = misc.eph.datetime.fromtimestamp(utime[0])
        date_str = transit_time.strftime('%Y_%m_%d_%H')
        filename = '/scratch/k/krs/connor/beam_fit' + src + date_str + '.hdf5'

        g = h5py.File(filename,'w')
        g.create_dataset('beam', data = beam_params)
        g.create_dataset('visibilities', data = vis_obj)
        g.create_dataset('RA', data = RA_src)
        g.close()

        print "Saved data in", filename
