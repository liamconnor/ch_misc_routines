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

outfile = '/scratch/k/krs/connor/B0329_pulse_phase_mpi.hdf5'
list = glob.glob('/scratch/k/krs/connor/chime/chime_data/20131210T060233Z/20131210T060233Z.h5.*')
list.sort()

chunk_length = 30
nchunks = len(list) / chunk_length

print "Total of %i files" % len(list)

jj = comm.rank

print "Starting chunk %i of %i" % (jj+1, nchunks)

data_arr, time, RA = misc.get_data(list[chunk_length*jj:chunk_length*(jj+1)])[1:]
data_arr = data_arr[:, 0, :]

time_int = 500 # Integrate in time for 500 samples
freq_int = 16 # Integrate over 16 freq bins

n_freq_bins = np.round( data_arr.shape[0] / freq_int )
n_time_bins = np.round( data_arr.shape[-1] / time_int )
n_phase_bins = 64
    
folded_arr = np.zeros([n_freq_bins, n_time_bins, n_phase_bins], np.complex128)

print "folded pulsar array has shape", folded_arr.shape
RC = chp.RFI_Clean(data_arr, time)
RC.frequency_clean()
    
for freq in range(n_freq_bins):
    print "freq", freq
    for tt in range(n_time_bins):
        folded_arr[freq, tt, :] = RC.fold_pulsar(p1, DM, nbins=n_phase_bins, \
                        start_chan=freq_int*freq, end_chan=freq_int*(freq+1), \
                        start_samp=time_int*tt, end_samp=time_int*(tt+1), f_ref=400.0)

fully = comm.gather(folded_arr, root=0)

if jj == 0:
    final_array = np.concatenate(fully, axis=1)
    print "Writing folded array to", outfile, "with shape:", final_array.shape
    f = h5py.File(outfile, 'w')
    f.create_dataset('folded_arr', data=final_array) 
    f.close()
