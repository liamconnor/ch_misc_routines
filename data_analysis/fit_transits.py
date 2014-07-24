import math
import numpy as np
import sys
from numpy.random import *

import scipy.optimize as so
import h5py
import argparse
import misc_data_io as misc
import fitting_modules as fm
import ch_util.ephemeris as eph

n_freq = 1024
n_corr = 136

parser = argparse.ArgumentParser(description="This programs tries to fit beam from point-source trans\
its.")
parser.add_argument("Data", help="Directory containing acquisition files.")
parser.add_argument("--Objects", help="Celestial objects to fit", default='All')
parser.add_argument("--minfile", help="Minfile number e.g. 0051", default="")
parser.add_argument("--maxfile", help="Maxfile number e.g. 0051", default="")
args = parser.parse_args()

files = np.str(args.Data) + '*h5.' + args.maxfile + '*'

print ""
print "Reading in Data:", files

Data, vis, utime, RA = misc.get_data(files)
print "RA range of data:", RA.min(),":",RA.max()

RA_sun = eph.transit_RA(eph.solar_transit(utime[0]))
sun_RA_low = RA_sun - 6
sun_RA_high = RA_sun + 6
print ""
print "Das sun was at: %f" % RA_sun

# Create a dictionary with each fitting object's information in the form: {"Obj": [RA_min, RA_max, Declination]}.
celestial_object = { "CasA": [344, 358, 58.83], "TauA": [77, 87, 83.6], "CygA": [297, 302, 40.73], "Sun": [sun_RA_low, sun_RA_high, 0]}

if args.Objects != "All":
    srcs2fit = [args.Objects]
else:
    srcs2fit = celestial_object.keys()

for src in srcs2fit:
    vis_obj = vis[:, :, ( RA > celestial_object[src][0] ) & ( RA < celestial_object[src][1])]
    RA_src = RA[ ( RA > celestial_object[src][0] ) & ( RA < celestial_object[src][1]) ]
    
    if len(RA_src) == 0:
        print src, "ain't here. Skipping..."
        print ""
    elif (RA_src[-1] - RA_src[0]) < 0.0:
        while (RA_src[-1] - RA_src[0]) < 0.0:
            RA_src = RA_src[:-5]
    else:
        print "Fitting", src
        print "Shape:", RA_src.shape

        beam_params = fm.beam_fit(vis_obj, RA_src, dec = celestial_object[src][2])

        transit_time = misc.eph.datetime.fromtimestamp(utime[0])
        date_str = transit_time.strftime('%Y_%m_%d')
        filename = '/scratch/k/krs/connor/beam_fit' + src + date_str + '.hdf5'

        g = h5py.File(filename,'w')
        g.create_dataset('beam', data = beam_params)
        g.create_dataset('visibilities', data = vis_obj)
        g.create_dataset('RA', data = RA_src)
        g.close()

        print "Saved data in", filename

