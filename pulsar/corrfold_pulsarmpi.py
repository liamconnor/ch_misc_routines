import os
import glob
import argparse

import numpy as np
import h5py

import ch_pulsar_analysis as chp
import misc_data_io as misc
import ch_util
import ch_util.ephemeris as eph

from mpi4py import MPI
comm = MPI.COMM_WORLD
print comm.rank, comm.size

parser = argparse.ArgumentParser(description="This script RFI-cleans, fringestops, and folds the pulsar data.")
parser.add_argument("data_dir", help="Directory with hdf5 data files")
parser.add_argument("pulsar", help="Name of pulsar e.g. B0329+54")
parser.add_argument("-n_phase_bins", help="Number of pulsar gates with which to fold", default=32, type=int)
parser.add_argument("-time_int", help="Number of samples to integrate over", default=1000, type=int)
parser.add_argument("-freq_int", help="Number of frequencies to integrate over", default=1, type=int)
parser.add_argument("-nfeeds", help="Number of feeds in acquisition", default=16, type=int)
parser.add_argument("-use_fpga", help="Use fpga counts instead of timestamps", default=1, type=int)
parser.add_argument("-add_tag", help="Add tag to outfile name to help identify data product", default='')
parser.add_argument("-nnodes", help='Number of nodes', default=30, type=int)
parser.add_argument("-chunksize", help='Number of files to read per node', default=10, type=int)
args = parser.parse_args()

sources = np.loadtxt('/home/k/krs/connor/code/ch_misc_routines/pulsar/sources2.txt', dtype=str)[1:]

src_ind = sources[sources[:,0]==args.pulsar][0]
RA_src, dec, DM, p1 = np.float(\
    src_ind[1]), np.float(src_ind[2]), np.float(src_ind[3]), np.float(src_ind[4])

nnodes = args.nnodes
file_chunk = args.chunksize

nfeeds = args.nfeeds
dat_name = args.data_dir[-16:]

list = glob.glob(args.data_dir + '/*h5*')
list.sort()

offs=0 # This is the file number offset, in case you only want to fold files n through m
list = list[offs:offs + file_chunk * nnodes]

nchunks = len(list) / file_chunk

print "Total of %i files" % len(list)

jj = comm.rank

print "Starting chunk %i of %i" % (jj+1, nchunks)
print "Getting", file_chunk*jj, ":", file_chunk*(jj+1)

x = 1
y = 0

feeds = np.arange(1, nfeeds)

# The following chooses only correlations between feed i and the 26m feeds
corrs_x = [misc.feed_map(i, x, nfeeds) for i in feeds if i!=x] \
#              + [misc.feed_map(i, y, nfeeds) for i in feeds if i!=x] 

corrs_auto = [misc.feed_map(i, i, nfeeds) for i in feeds]

# Or you can just select random correlation products
corrs = corrs_x + corrs_auto

ncorr = len(corrs)   
corrs.sort()
    
if jj == 0:
    print "RA, dec, DM, period:", RA_src, dec, DM, p1
    print "Using correlations", corrs

data_reader_obj = ch_util.andata.Reader(list[file_chunk*jj:file_chunk*(jj+1)])
data_reader_obj.prod_sel = corrs

data_obj = data_reader_obj.read()
data_arr = data_obj.vis
time = data_obj.timestamp

if jj == 0:
    t0 = time[0]

RA = eph.transit_RA(time)
fpga_count = data_obj.index_map['time']['fpga_count']

ntimes = len(time)

time_int = args.time_int
freq_int = args.freq_int 

outdir = '/scratch/k/krs/connor/chime/calibration/' 

if args.use_fpga==1:

    dt = 2048. / 800e6 #* 4.0 # 
    # THIS 4 IS ONLY FOR AUGUST 22ND WHOSE TIMESTAMPS ARE BAD
    time = (fpga_count) * dt 

    if jj==0:
        print "We're going with the fpga counts"
        print "Median del t", np.median(np.diff(time))


n_freq_bins = np.round(data_arr.shape[0] / freq_int)
n_time_bins = np.round(data_arr.shape[-1] / time_int)
ngate = args.n_phase_bins

RC = chp.RFI_Clean(data_arr, time)

folded_arr, icount = RC.fold(DM, p1, ngate=ngate, ntrebin=time_int)

print "Done folding"

count=0

for letter in args.data_dir:
    count += 1
    if letter == '2':
        filename = args.data_dir[count-1:count+15]
        break

outdir = outdir + filename + '/'

if os.path.isdir(outdir):
    pass
else:
    os.mkdir(outdir)

times_actually_full = comm.gather(time, root=0) 

for freq in range(n_freq_bins):

    folded_corr = comm.gather(folded_arr[freq, :, np.newaxis], root=0)        
    icount_full = comm.gather(icount[freq], root=0)

    if jj == 0:
        print "Done gathering arrays for freq", freq, folded_corr[0].shape, len(folded_corr)
        final_arr = np.concatenate(folded_corr, axis=1).reshape(ncorr, -1, ngate)
        final_icount = np.concatenate(icount_full, axis=1)

        times = np.concatenate(times_actually_full)

        outfile = outdir + args.pulsar + filename + \
            np.str(freq_int) + np.str(args.add_tag) + np.str(freq) + '.hdf5'

        if os.path.isfile(outfile):
            os.remove(outfile)

        print "Writing output to", outfile

        times += t0

        f = h5py.File(outfile, 'w')
        f.create_dataset('folded_arr', data=final_arr)
        f.create_dataset('icount', data=final_icount)
        f.create_dataset('times', data=times)
        f.attrs.create('corrs', data=corrs)
        f.close()


