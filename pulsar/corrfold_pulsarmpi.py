import os
import glob
import argparse

import numpy as np
import h5py

import ch_pulsar_analysis as chp
import misc_data_io as misc

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
parser.add_argument("-div_autos", help='Divide by geometric mean of autos.', default=1, type=int)
parser.add_argument("-use_chime_autos", help='use only ch-ch autocorrelations', default=0, type=int)
args = parser.parse_args()

sources = np.loadtxt('/home/k/krs/connor/code/ch_misc_routines/pulsar/sources2.txt', dtype=str)[1:]

src_ind = sources[sources[:,0]==args.pulsar][0]
RA_src, dec, DM, p1 = np.float(src_ind[1]), np.float(src_ind[2]), np.float(src_ind[3]), np.float(src_ind[4])

nnodes = args.nnodes
file_chunk = args.chunksize

nfeeds = args.nfeeds
dat_name = args.data_dir[-16:]

list = glob.glob(args.data_dir + '/*h5*')
list.sort()

offs=215
list = list[offs:offs + file_chunk * nnodes]

nchunks = len(list) / file_chunk

print "Total of %i files" % len(list)

jj = comm.rank

print "Starting chunk %i of %i" % (jj+1, nchunks)
print "Getting", file_chunk*jj, ":", file_chunk*(jj+1)

x=7
y=3

feeds = np.arange(nfeeds)
nfeeds = len(feeds)
feeds = [3, 7]
corrs = [misc.feed_map(i, x, nfeeds) for i in feeds] + [misc.feed_map(i, y, nfeeds) for i in feeds] 

corrs_auto = [misc.feed_map(i, i, nfeeds) for i in feeds]
corrs = range(nfeeds * (nfeeds+1)/2)

ncorr = len(corrs)   

if args.use_chime_autos:
    feeds = np.arange(nfeeds)
    corrs_auto = [misc.feed_map(i, i, nfeeds) for i in feeds]
    corrs = corrs_auto
    # Do a run with only xy autos
    xcorr = [4, 5, 6, 7, 8, 9, 10, 11]
    ycorr = [0, 1, 2, 3, 12, 13, 14, 15]

    corrs = [misc.feed_map(xcorr[i], ycorr[i], nfeeds) for i in range(len(xcorr))]
    corrs_auto = [misc.feed_map(i, i, nfeeds) for i in xcorr] + \
                 [misc.feed_map(i, i, nfeeds) for i in ycorr]

    corrs = xx + yy
    corrs_auto = xauto + yauto

    corrs_auto = [misc.feed_map(i, i, 16) for i in range(16)]
    corrs = range(1, 16) + corrs_auto

    ncorr = len(corrs)
    
if jj==0:
    print "RA, dec, DM, period:", RA_src, dec, DM, p1
    print "Using correlations", corrs
    print "with autos:", corrs_auto    

data_arr_full, time_full, RA, fpga_count = misc.get_data(list[file_chunk*jj:file_chunk*(jj+1)])[1:]
data_arr = data_arr_full[:, corrs]

ntimes = len(time_full)
time  = time_full

time_int = args.time_int
freq_int = args.freq_int 

outdir = '/scratch/k/krs/connor/chime/calibration/' 

if args.use_fpga==1:
    dt = 2048. / 800e6 #* 4.0 # THIS IS ONLY FOR AUGUST 22ND WHOSE TIMESTAMPS ARE BAD
    time = (fpga_count) * dt 
    if jj==0:
        print "We're going with the fpga counts"
        print "Median del t", np.median(np.diff(time))

n_freq_bins = np.round(data_arr.shape[0] / freq_int)
n_time_bins = np.round(data_arr.shape[-1] / time_int)
ngate = args.n_phase_bins

#folded_arr = np.zeros([n_freq_bins, ncorr, n_time_bins, n_phase_bins], np.complex128)

RC = chp.RFI_Clean(data_arr, time)
#RC.corrs = corrs
#RC.frequency_clean()

folded_arr, icount = RC.fold(data_arr, time, DM, p1, ngate=ngate, ntrebin=time_int)

folded_arr = np.concatenate((folded_arr1, folded_arr2), axis=0)
print "Done folding"

"""
for freq in range(n_freq_bins):
    if jj==0:
        print "Folding freq %i" % freq 
    for tt in range(n_time_bins):
        folded_arr[freq, :, tt, :] = RC.fold_pulsar(p1, DM, nbins=ngate, \
                    start_chan=freq_int*freq, end_chan=freq_int*(freq+1), start_samp=time_int*tt, end_samp=time_int*(tt+1), f_ref=400.0)
"""

count=0

for letter in args.data_dir:
    count+=1
    if letter=='2':
        filename=args.data_dir[count-1:count+15]
        break

outdir = outdir + filename + '/'

if os.path.isdir(outdir):
    pass
else:
    os.mkdir(outdir)

times_actually_full = comm.gather(time, root=0)

freq_full = np.linspace(800, 400, 1024)

for freq in range(n_freq_bins):

    folded_corr = comm.gather(folded_arr[freq, :, np.newaxis], root=0)        
    icount_full = comm.gather(icount[freq], root=0)
    auto_arr = comm.gather(data_arr_full[freq, corrs_auto], root=0)

    if jj == 0:
        print "Done gathering arrays for freq", freq, folded_corr[0].shape, len(folded_corr)
        final_arr = np.concatenate(folded_corr, axis=1).reshape(ncorr, -1, ngate)
        final_icount = np.concatenate(icount_full, axis=1)

        if args.div_autos == 1:

            auto_arr = np.concatenate(auto_arr, axis=1).mean(-1)[:, np.newaxis, np.newaxis]
            print "Dividing by autos post"
  
            final_arr[:nfeeds] /= np.sqrt(abs(auto_arr) * abs(auto_arr[x, np.newaxis])) * freq_full[freq]**(0.8)

            final_arr[nfeeds:] /= np.sqrt(abs(auto_arr) * abs(auto_arr[y, np.newaxis])) * freq_full[freq]**(0.8)

        times = np.concatenate(times_actually_full)

        outfile = outdir + args.pulsar + filename + np.str(freq_int) + np.str(args.add_tag) + np.str(freq) + '.hdf5'

        if os.path.isfile(outfile):
            os.remove(outfile)

        print "Writing output to", outfile

        f = h5py.File(outfile, 'w')
        f.create_dataset('folded_arr', data=final_arr)
        f.create_dataset('icount', data=final_icount)
        f.create_dataset('times', data=times)
        f.close()


