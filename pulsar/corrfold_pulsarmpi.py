import numpy as np
import ch_pulsar_analysis as chp
import h5py
import misc_data_io as misc
import glob
import os
import argparse

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

RA_src, dec, DM, p1 = np.float(sources[sources[:,0]==args.pulsar][0][1]), np.float(sources[sources[:,0]==args.pulsar][0][2]), np.float(sources[sources[:,0]==args.pulsar][0][3]),\
    np.float(sources[sources[:,0]==args.pulsar][0][4])

nnodes = args.nnodes
file_chunk = args.chunksize

nfeeds = args.nfeeds
dat_name = args.data_dir[-16:]

list = glob.glob(args.data_dir + '/*h5*')
list.sort()
list = list[0:0 + file_chunk * nnodes]

nchunks = len(list) / file_chunk

print "Total of %i files" % len(list)

jj = comm.rank

print "Starting chunk %i of %i" % (jj+1, nchunks)
print "Getting", file_chunk*jj, ":", file_chunk*(jj+1)

corrs = [misc.feed_map(i, 3, nfeeds) for i in range(nfeeds)] + [misc.feed_map(i, 7, nfeeds) for i in range(nfeeds)] 
corrs_auto = [misc.feed_map(i, i, nfeeds) for i in range(nfeeds)]
ncorr = len(corrs)    


if jj==0:
    print "RA, dec, DM, period:", RA_src, dec, DM, p1
    print "Using correlations", corrs
    print "with autos:", corrs_auto
    

data_arr_full, time_full, RA, fpga_count = misc.get_data(list[file_chunk*jj:file_chunk*(jj+1)])[1:]
data_arr = data_arr_full[:, corrs]

print "Now dividing by autos"

print np.sqrt((abs(data_arr_full[:, corrs_auto])**2) * abs(data_arr_full[:, corrs_auto[3], np.newaxis])**2).shape

data_arr[:, :nfeeds] /= np.sqrt((abs(data_arr_full[:, corrs_auto])**2) * abs(data_arr_full[:, corrs_auto[3], np.newaxis])**2)
data_arr[:, nfeeds:] /= np.sqrt((abs(data_arr_full[:, corrs_auto])**2) * abs(data_arr_full[:, corrs_auto[7], np.newaxis])**2)


ntimes = len(time_full)
time  = time_full

time_int = args.time_int
freq_int = args.freq_int 

outdir = '/scratch/k/krs/connor/chime/calibration/' 

if args.use_fpga==1:
    dt = 2048. / 800e6
    time = (fpga_count) * dt
    print "We're going with the fpga counts"


n_freq_bins = np.round( data_arr.shape[0] / freq_int )
n_time_bins = np.round( data_arr.shape[-1] / time_int )
n_phase_bins = args.n_phase_bins

folded_arr = np.zeros([n_freq_bins, ncorr, n_time_bins, n_phase_bins], np.complex128)

print "folded pulsar array", jj, "has shape", folded_arr.shape

RC = chp.RFI_Clean(data_arr, time)
RC.dec = dec
RC.RA = RA
RC.RA_src = np.deg2rad(RA_src)
print RC.data.shape, "ZERO"
RC.corrs = corrs#[: 2 * ncorr / 3] # Need only to fold the CHIME/26m correlations. 
print RC.data.shape, "TWO"
RC.frequency_clean()
RC.fringestop() 

for freq in range(n_freq_bins):
    print "Folding freq %i" % freq 
    for tt in range(n_time_bins):
        folded_arr[freq, :, tt, :] = RC.fold_pulsar(p1, DM, nbins=n_phase_bins, \
                    start_chan=freq_int*freq, end_chan=freq_int*(freq+1), start_samp=time_int*tt, end_samp=time_int*(tt+1), f_ref=400.0)

fullie = []
final_list = []

count=0
print args.data_dir
for letter in args.data_dir:
    count+=1
    if letter=='2':
        filename=args.data_dir[count-1:count+15]
        print filename
        break

outdir = outdir + filename + '/'

if os.path.isdir(outdir):
    pass
else:
    os.mkdir(outdir)

times_actually_full = comm.gather(time, root=0)

for freq in range(n_freq_bins):

    folded_corr = comm.gather(folded_arr[freq, :, np.newaxis], root=0)
  
    if jj == 0:
        print "Done gathering arrays for freq", freq, folded_corr[0].shape, len(folded_corr)
#        final_list.append(np.concatenate(folded_corr, axis=1))
        final_arr = np.concatenate(folded_corr, axis=1).reshape(ncorr, -1, args.n_phase_bins)
        times = np.concatenate(times_actually_full)
        outfile = outdir + args.pulsar + filename + np.str(freq_int) + np.str(args.add_tag) + np.str(freq) + '.hdf5'

        if os.path.isfile(outfile):
            os.remove(outfile)

        print "Writing output to", outfile, "array has shape"#, folded_corr.shape

        f = h5py.File(outfile, 'w')
        f.create_dataset('folded_arr', data=final_arr)
        f.create_dataset('times', data=times)
        f.close()


