import numpy as np
import ch_pulsar_analysis as chp
import h5py
import misc_data_io as misc
import glob
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
print comm.rank, comm.size

DM = 26.833 # B0329 dispersion measure
p1 = 0.7145817552986237 # B0329 period
#p1 = 0.71457950245055065

dec = 54.57876944444

nnodes = 64
file_chunk = 8

outdir = '/scratch/k/krs/connor/pulsar_analysis/'

parser = argparse.ArgumentParser(description="This script RFI-cleans, fringestops, and folds the pulsar data.")
parser.add_argument("data_dir", help="Directory with hdf5 data files")
parser.add_argument("pulsar", help="Name of pulsar e.g. B0329")
parser.add_argument("--n_phase_bins", help="Number of pulsar gates with which to fold", default=64)
parser.add_argument("--time_int", help="Number of samples to integrate over", default=1000)
parser.add_argument("--freq_int", help="Number of frequencies to integrate over", default=1)
parser.add_argument("--ncorr", help="Number of correlations to include", default=36)
args = parser.parse_args()

sources = np.loadtxt('sources.txt', dtype=str)[1:]
dec = np.float(sources[sources[0]==args.pulsar][0][2])

ncorr = args.ncorr
dat_name = args.data_dir[-16:]

list = glob.glob(args.data_dir + '/*h5')
list.sort()
list = list[:file_chunk * nnodes]

nchunks = len(list) / file_chunk

print "Total of %i files" % len(list)

jj = comm.rank

print "Starting chunk %i of %i" % (jj+1, nchunks)
print "Getting", file_chunk*jj, ":", file_chunk*(jj+1)

data_arr, time_full, RA = misc.get_data(list[file_chunk*jj:file_chunk*(jj+1)])[1:]
data_arr = data_arr[:, :ncorr, :]

ntimes = len(time_full)
time = time_full

time_int = args.time_int
freq_int = args.freq_int 

"""g = h5py.File('/scratch/k/krs/connor/psr_fpga.hdf5','r')
fpga = g['fpga'][:]
times = (fpga - fpga[0]) * 0.01000 / 3906.0
time = times[jj * ntimes : (jj+1) * ntimes]
"""

n_freq_bins = np.round( data_arr.shape[0] / freq_int )
n_time_bins = np.round( data_arr.shape[-1] / time_int )
n_phase_bins = args.n_phase_bins
    
folded_arr = np.zeros([n_freq_bins, ncorr, n_time_bins, n_phase_bins], np.complex128)

print "folded pulsar array has shape", folded_arr.shape

RC = chp.RFI_Clean(data_arr, time)
RC.dec = dec
RC.RA = RA
RC.frequency_clean(threshold=1e6)
RC.fringe_stop() 

for freq in range(n_freq_bins):
    print "Folding freq %i" % freq 
    for tt in range(n_time_bins):
        folded_arr[freq, :, tt, :] = RC.fold_pulsar(p1, DM, nbins=n_phase_bins, \
                    start_chan=freq_int*freq, end_chan=freq_int*(freq+1), start_samp=time_int*tt, end_samp=time_int*(tt+1), f_ref=400.0)

fullie = []
final_list = []

if os.path.isdir(outdir + dat_name):
    pass
else
    os.mkdir(outdir + dat_name)

for corr in range(ncorr):

    folded_corr = comm.gather(folded_arr[:, np.newaxis, corr, :, :], root=0)
    if jj == 0:
        print "Done gathering arrays for corr", corr
        final_list.append(np.concatenate(folded_corr, axis=2))

if jj==0:
    print len(final_list), final_list[0].shape
    final_array = np.concatenate(final_list, axis=1)
    outfile = outdir + dat_name + '/' + dat_name + 'folded_array.hdf5'
    print "Writing folded array to", outfile, "with shape:", final_array.shape

    f = h5py.File(outfile, 'w')
    f.create_dataset('folded_arr', data=final_array) 
    f.close()

