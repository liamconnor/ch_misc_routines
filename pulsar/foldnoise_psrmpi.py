import os
import glob
import argparse

import numpy as np
import h5py

import ch_pulsar_analysis as chp
import misc_data_io as misc
import ch_util.andata
import ch_util.ephemeris as eph

from mpi4py import MPI
comm = MPI.COMM_WORLD
print comm.rank, comm.size

def inj_fake_noise(data, times, cad=5):
    t_ns, p0 = np.linspace(times[0], times[-1], 
              ((times[-1] - times[0]) / cad).astype(int), retstep=True)
    tt = times.repeat(len(t_ns)).reshape(-1, len(t_ns))

    ind_ns = abs(tt - t_ns).argmin(axis=0)
    
    for corr in range(data.shape[1]):
        data[:, corr, ind_ns] += np.median(data[:, corr], axis=-1)[:, np.newaxis]

    return data, p0


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

inj_noise = True

sources = np.loadtxt('/home/k/krs/connor/code/ch_misc_routines/pulsar/sources2.txt', dtype=str)[1:]

src_ind = sources[sources[:, 0]==args.pulsar][0]
RA_src, dec, DM, p1 = np.float(src_ind[1]), \
    np.float(src_ind[2]), np.float(src_ind[3]), np.float(src_ind[4])

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
    corrs = corrs_auto
    ncorr = len(corrs)
    
if jj==0:
    print "RA, dec, DM, period:", RA_src, dec, DM, p1
    print "Using correlations", corrs
    print "with autos:", corrs_auto    

feeds_ns = [45, 91]

Reader_obj = ch_util.andata.Reader(list[file_chunk*jj:file_chunk*(jj+1)])
Reader_obj.prod_sel = feeds_ns

time = Reader_obj.time
RA = eph.transit_RA(time)

data_obj = Reader_obj.read()

fpga_count = data_obj['timestamp_fpga_count'][:]
data_arr = data_obj.vis

ntimes = len(time)

time_int = args.time_int
freq_int = args.freq_int 
ncorr = data_arr.shape[1]

outdir = '/scratch/k/krs/connor/chime/calibration/' 

if args.use_fpga==1:
    dt = 2048. / 800e6 * 4.0 # THIS IS ONLY FOR AUGUST 22ND WHOSE TIMESTAMPS ARE BAD
    time = (fpga_count) * dt 
    if jj==0:
        print "We're going with the fpga counts"
        print "Median del t", np.median(np.diff(time))

n_freq_bins = np.round(data_arr.shape[0] / freq_int)
n_time_bins = np.round(data_arr.shape[-1] / time_int)
ngate = args.n_phase_bins


RC = chp.RFI_Clean(data_arr, time)
#RC.corrs = corrs
#RC.frequency_clean()

if inj_noise:
    data_arr, p_ns = inj_fake_noise(data_arr, time, cad=1.0000)
    
folded_arr, icount = RC.fold(data_arr, time, 0.0, p_ns, ngate=ngate, ntrebin=time_int)

print "Done folding"

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

time_full = comm.gather(time, root=0)

freq_full = np.linspace(800, 400, 1024)

for freq in range(n_freq_bins):

    folded_corr = comm.gather(folded_arr[freq, :, np.newaxis], root=0)        
    icount_full = comm.gather(icount[freq], root=0)

    if jj == 0:
        print "Done gathering arrays for freq", freq, folded_corr[0].shape, len(folded_corr)
        final_arr = np.concatenate(folded_corr, axis=1).reshape(ncorr, -1, ngate)
        final_icount = np.concatenate(icount_full, axis=1)

        times = np.concatenate(time_full)

        outfile = outdir + args.pulsar + filename + np.str(freq_int) + np.str(args.add_tag) + np.str(freq) + '.hdf5'

        if os.path.isfile(outfile):
            os.remove(outfile)

        print "Writing output to", outfile

        f = h5py.File(outfile, 'w')
        f.create_dataset('folded_arr', data=final_arr)
        f.create_dataset('icount', data=final_icount)
        f.create_dataset('times', data=times)
        f.close()


