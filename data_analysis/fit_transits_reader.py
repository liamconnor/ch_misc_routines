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

parser = argparse.ArgumentParser(description="This programs tries to fit beam from point-source trans\
its.")
parser.add_argument("data_dir", help="Directory containing acquisition directories.")
parser.add_argument("start_date", help="First day to search for transit in. Should be six digits YYYYMMDD", default=None)
parser.add_argument("stop_date", help="Last day to search for transit in. Should be six digits YYYYMMDD", default=None)
parser.add_argument("--src", help="Source whose transit will be fit", default='CasA')
parser.add_argument("--outdir", help="Directory output file gets written to", default="/home/liam/")
parser.add_argument("--product_set", help="Which subset of correlations to fit (e.g. autos)", default=None)
args = parser.parse_args()

src = args.src

# Create a dictionary with each fitting object's information in the form: {"Obj": [RA_min, RA_max, Declination]}.
celestial_object = { "CasA": [344, 358, 58.83, ch_util.ephemeris.CasA], "TauA": [77, 87, 83.6, ch_util.ephemeris.TauA], "CygA": [297, 302, 40.73]}

st = args.start_date
end = args.stop_date

f = data_index.Finder()
f.set_time_range(datetime(int(st[:4]), int(st[4:6]), int(st[-2:])), datetime(int(end[:4]), int(end[4:6]), int(end[-2:])))
f.include_transits(celestial_object[src][3], time_delta=5000)
file_list = f.get_results()[0][0]
print "Found transit in:", file_list

for ii in range(len(file_list)):
    file_list[ii] = args.data_dir + file_list[ii]


print ""
print "Reading in Data:", file_list
print ""

dataobj = ch_util.andata.Reader(file_list)
X = dataobj.read()
vis, times, nprod = X.vis, X.timestamp, X.nprod
RA = ch_util.ephemeris.transit_RA(times)
nant = int((-1 + (1 + 4 * 2 * nprod)**0.5) / 2)

m = 3
if args.product_set=='autos':
    corrs = [misc.feed_map(i, i, nant) for i in range(nant)]
elif args.product_set=='26m':
    corrs = [misc.feed_map(i, m, nant) for i in range(nant)]
else:
    corrs = [0]

print "RA range of data:", RA.min(),":",RA.max()
print "for correlations", corrs

beam_params = fm.beam_fit(vis, RA, dec = celestial_object[src][2], correlations=corrs) # This is where the actual fit is done.

transit_time = ch_util.ephemeris.datetime.fromtimestamp(times[0])
date_str = transit_time.strftime('%Y_%m_%d')
filename = args.outdir + src + date_str + '.hdf5'

print "Writing beam parameters file to %s" % filename

g = h5py.File(filename,'w')
g.create_dataset('beam', data = beam_params)
g.create_dataset('visibilities', data = vis)
g.create_dataset('RA', data = RA)
g.close()

