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

outdir = '/scratch/k/krs/connor/chime/calibration/'

parser = argparse.ArgumentParser(description="This script RFI-cleans, fringestops, and folds the pulsar data.")
parser.add_argument("data_dir", help="Directory with hdf5 data files")
parser.add_argument("pulsar", help="Name of pulsar e.g. B0329+54")
parser.add_argument("--n_phase_bins", help="Number of pulsar gates with which to fold", default=32, type=int)
parser.add_argument("--time_int", help="Number of samples to integrate over", default=1000, type=int)
parser.add_argument("--freq_int", help="Number of frequencies to integrate over", default=1, type=int)
parser.add_argument("--ncorr", help="Number of correlations to include", default=36, type=int)
parser.add_argument("--use_fpga", help="Use fpga counts instead of timestamps", default=0, type=int)
parser.add_argument("--add_tag", help="Add tag to outfile name to help identify data product", default='')
parser.add_argument("--nnodes", help='Number of nodes', default=30, type=int)
parser.add_argument("--chunksize", help='Number of files to read per node', default=10, type=int)
args = parser.parse_args()

sources = np.loadtxt('/home/k/krs/connor/code/ch_misc_routines/pulsar/sources2.txt', dtype=str)[1:]

RA_src, dec, DM, p1 = np.float(sources[sources[:,0]==args.pulsar][0][1]), np.float(sources[sources[:,0]==args.pulsar][0][2]), np.float(sources[sources[:,0]==args.pulsar][0][3]),\
    np.float(sources[sources[:,0]==args.pulsar][0][4])

nnodes = args.nnodes
file_chunk = args.chunksize

ncorr = args.ncorr
dat_name = args.data_dir[-16:]

list = glob.glob(args.data_dir + '/*h5*')
list.sort()
list = list[0:0 + file_chunk * nnodes]

nchunks = len(list) / file_chunk

print "Total of %i files" % len(list)

jj = comm.rank

print "Starting chunk %i of %i" % (jj+1, nchunks)
print "Getting", file_chunk*jj, ":", file_chunk*(jj+1)

corrs = [misc.feed_map(i, 3, 16) for i in range(16)] #+ [misc.feed_map(i, 3, 16) for i in range(16)] 

"""
#corrs = [
3 ,
18 ,
32 ,
45 ,
46 ,
47 ,
48 ,
49 ,
50 ,
51 ,
52 ,
53 ,
54 ,
55 ,
56 ,
57 ,
7 ,
22 ,
36 ,
49 ,
61 ,
72 ,
82 ,
91 ,
92 ,
93 ,
94 ,
95 ,
96 ,
97 ,
98 ,
99 ,
]
"""
#corrs = [0, 16, 7, 45, 91]
corrs = [35, 57, 45, 91]

data_arr, time_full, RA, fpga_count = misc.get_data(list[file_chunk*jj:file_chunk*(jj+1)])[1:]
data_arr = data_arr[:, corrs, :]

ntimes = len(time_full)
time  = time_full

time_int = args.time_int
freq_int = args.freq_int 

fpga_tag = ''

if args.use_fpga==1:
    time = (fpga_count - fpga_count[0]) * (np.diff(time_full)[0]) / np.diff(fpga_count)[0]
    print "We're going with the fpga counts:", np.median(np.diff(fpga_count))
    fpga_tag = 'fpga'

print "The median time diff is:", np.median(np.diff(time))

n_freq_bins = np.round( data_arr.shape[0] / freq_int )
n_time_bins = np.round( data_arr.shape[-1] / time_int )
n_phase_bins = args.n_phase_bins

ncorr = len(corrs)    
folded_arr = np.zeros([n_freq_bins, ncorr, n_time_bins, n_phase_bins], np.complex128)

print "folded pulsar array", jj,"has shape", folded_arr.shape

RC = chp.RFI_Clean(data_arr, time)
RC.dec = dec
RC.RA = RA
RC.RA_src = 0.934041310805 # Obvi need to change this.
RC.corrs = corrs
RC.frequency_clean()
RC.fringestop() 

for freq in range(n_freq_bins):
    print "Folding freq %i" % freq 
    for tt in range(n_time_bins):
        folded_arr[freq, :, tt, :] = RC.fold_pulsar(p1, DM, nbins=n_phase_bins, \
                    start_chan=freq_int*freq, end_chan=freq_int*(freq+1), start_samp=time_int*tt, end_samp=time_int*(tt+1), f_ref=400.0)

fullie = []
final_list = []

if os.path.isdir(outdir + dat_name):
    pass
else:
    os.mkdir(outdir + dat_name)

for corr in range(ncorr):

    folded_corr = comm.gather(folded_arr[:, np.newaxis, corr, :, :], root=0)
    if jj == 0:
        print "Done gathering arrays for corr", corr
        final_list.append(np.concatenate(folded_corr, axis=2))

times_actually_full = comm.gather(time_full, root=0)

if jj==0:
    final_array = np.concatenate(final_list, axis=1)
    times = np.concatenate(times_actually_full)
    print times.shape,"IS DA SHAPE OF DA TIMES"
    
    count=0
    print args.data_dir
    for letter in args.data_dir:
        count+=1
        if letter=='2':
            filename=args.data_dir[count-1:count+15]
            print filename
            break

    outfile = outdir + args.pulsar + filename + np.str(freq_int) + fpga_tag + np.str(args.add_tag) + '.hdf5'
    
    if os.path.isfile(outfile):
        os.remove(outfile)
    
    print "Writing output to", outfile
    f = h5py.File(outfile, 'w')
    f.create_dataset('folded_arr', data=final_array) 
    f.create_dataset('times', data=times)
    f.close()

