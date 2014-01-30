import numpy as np
import ch_pulsar_analysis as chp
import h5py
import misc_data_io as misc
import glob

from mpi4py import MPI
comm = MPI.COMM_WORLD
print comm.rank, comm.size

DM = 26.833 # B0329 dispersion measure
p1 = 0.7145817552986237 # B0329 period
ncorr = 36

outdir = '/scratch/k/krs/connor/'
list = glob.glob('/scratch/k/krs/connor/chime/chime_data/20131210T060233Z/20131210T060233Z.h5.*')
list.sort()
list = list[:31*16]

chunk_length = 31
nchunks = len(list) / chunk_length

print "Total of %i files" % len(list)

jj = comm.rank

print "Starting chunk %i of %i" % (jj+1, nchunks)
print "Getting", chunk_length*jj,":",chunk_length*(jj+1)
data_arr, time, RA = misc.get_data(list[chunk_length*jj:chunk_length*(jj+1)])[1:]
#data_arr = data_arr[:, 8, :]
ntimes = len(time)

g = h5py.File('/scratch/k/krs/connor/psr_fpga.hdf5','r')
fpga = g['fpga'][:]
times = (fpga - fpga[0]) * 0.01000 / 3906.0

time = times[jj * ntimes : (jj+1) * ntimes]
print time.shape, data_arr.shape[-1]

time_int = 500 # Integrate in time for 500 samples
freq_int = 16 # Integrate over 16 freq bins

n_freq_bins = np.round( data_arr.shape[0] / freq_int )
n_time_bins = np.round( data_arr.shape[-1] / time_int )
n_phase_bins = 64
    
folded_arr = np.zeros([n_freq_bins, n_time_bins, n_phase_bins], np.complex128)

print "folded pulsar array has shape", folded_arr.shape

for corr in range(ncorr):
    print "Correlation product %i" % corr
    RC = chp.RFI_Clean(data_arr[:, corr, :], time)
    RC.frequency_clean()
    
    for freq in range(n_freq_bins):
        for tt in range(n_time_bins):
            folded_arr[freq, tt, :] = RC.fold_pulsar(p1, DM, nbins=n_phase_bins, \
                        start_chan=freq_int*freq, end_chan=freq_int*(freq+1), \
                        start_samp=time_int*tt, end_samp=time_int*(tt+1), f_ref=400.0)

    fully = comm.gather(folded_arr, root=0)
    print "Done gathering arrays"
    if jj == 0:
        final_array = np.concatenate(fully, axis=1)
        outfile = outdir + 'fpga_mpi_psr_phase' + np.str(corr) + '.hdf5'
        print "Writing folded array to", outfile, "with shape:", final_array.shape
        f = h5py.File(outfile, 'w')
        f.create_dataset('folded_arr', data=final_array) 
        f.close()
