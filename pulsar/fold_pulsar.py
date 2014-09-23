import numpy as np
import ch_pulsar_analysis as chp
import h5py
import misc_data_io as misc
import glob
import os
import argparse

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
parser.add_argument("--nfiles", help='Number of nodes', default=10, type=int)
args = parser.parse_args()

sources = np.loadtxt('/home/k/krs/connor/code/ch_misc_routines/pulsar/sources2.txt', dtype=str)[1:]

RA_src, dec, DM, p1 = np.float(sources[sources[:,0]==args.pulsar][0][1]), np.float(sources[sources[:,0]==args.pulsar][0][2]), np.float(sources[sources[:,0]==args.pulsar][0][3]),\
    np.float(sources[sources[:,0]==args.pulsar][0][4])

ncorr = args.ncorr
dat_name = args.data_dir[-16:]

list = glob.glob(args.data_dir + '*h5*')
list.sort()
list = list[0:0 + args.nfiles]

print "Total of %i files" % len(list)

corrs = [misc.feed_map(i, i, 16) for i in range(16)] #+ [misc.feed_map(i, 3, 16) for i in range(16)] 
corrs=[45,91]

data_arr, time_full, RA, fpga_count = misc.get_data(list)[1:]
data_arr = data_arr[:, corrs, :]

ntimes = len(time_full)
time  = time_full

time_int = args.time_int
freq_int = args.freq_int 

#fpga_count=[]
#for file in list:
#    f = h5py.File(file, 'r')
#    fpga_count.append(f['timestamp'].value['fpga_count'])

#print "lenfpga", len(fpga_count)
#fpga_count = np.concatenate(fpga_count)
#time = (fpga_count - fpga_count[0]) * (np.diff(time_full)[0]) / np.diff(fpga_count)[0]
#print fpga_count.shape

if args.use_fpga==1:
    dt = 2048. / 800e6
    time = (fpga_count - fpga_count[0]) * dt
    print "We're going with the fpga counts:", np.median(np.diff(fpga_count))
    fpga_tag = 'fpga'

print "The median time diff is:", np.median(np.diff(time))

n_freq_bins = np.round( data_arr.shape[0] / freq_int )
n_time_bins = np.round( data_arr.shape[-1] / time_int )
n_phase_bins = args.n_phase_bins

ncorr = len(corrs)    
folded_arr = np.zeros([n_freq_bins, ncorr, n_time_bins, n_phase_bins], np.complex128)

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

#if os.path.isdir(outdir + dat_name):
#    pass
#else:
#    os.mkdir(outdir + dat_name)

outfile = '/home/k/krs/connor/out_test' + args.add_tag + '.hdf5'
print "Writing output to", outfile
del_t = (time[-1] - time[0]) / 60.0
print "Folded %f minutes of data" % del_t
f = h5py.File(outfile, 'w')
f.create_dataset('folded_arr', data=folded_arr)
f.create_dataset('times', data=time)
f.close()

