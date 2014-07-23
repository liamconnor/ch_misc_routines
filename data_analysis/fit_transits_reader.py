import math
import numpy as np
import sys
from numpy.random import *

import scipy.optimize as so
import h5py
import argparse
import misc_data_io as misc
import fitting_modules as fm
import ch_util
from ch_util import data_index
from datetime import datetime

n_freq = 1024
n_corr = 136

parser = argparse.ArgumentParser(description="This programs tries to fit beam from point-source trans\
its.")
parser.add_argument("data_dir", help="Directory containing acquisition directories.")
parser.add_argument("--src", help="Celestial objects to fit", default='All')
parser.add_argument("--minfile", help="Minfile number e.g. 0051", default="")
parser.add_argument("--maxfile", help="Maxfile number e.g. 0051", default="")
args = parser.parse_args()

src = args.src

# Create a dictionary with each fitting object's information in the form: {"Obj": [RA_min, RA_max, Declination]}.
celestial_object = { "CasA": [344, 358, 58.83, ch_util.ephemeris.CasA], "TauA": [77, 87, 83.6, ch_util.ephemeris.TauA], "CygA": [297, 302, 40.73]}

f = data_index.Finder()
f.set_time_range(datetime(2014,03,20), datetime(2014,03,22))
f.include_transits(celestial_object[src][3], time_delta=3600)
file_list = f.get_results()[0][0]

for ii in range(len(file_list)):
    file_list[ii] = args.data_dir + file_list[ii]

dataobj = ch_util.andata.Reader(file_list)
X = dataobj.read()
vis, times = X.vis, X.timestamp
RA = ch_util.ephemeris.transit_RA(times)

print ""
print "Reading in Data:", file
print ""
print "RA range of data:", RA.min(),":",RA.max()

beam_params = fm.beam_fit(vis, RA, dec = celestial_object[src][2])

transit_time = ch_util.ephemeris.datetime.fromtimestamp(times[0])
date_str = transit_time.strftime('%Y_%m_%d')
filename = '/scratch/k/krs/connor/beam_fit' + src + date_str + '.hdf5'

g = h5py.File(filename,'w')
g.create_dataset('beam', data = beam_params)
g.create_dataset('visibilities', data = vis_obj)
g.create_dataset('RA', data = RA_src)
g.close()

