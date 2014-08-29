import numpy as np
import h5py
import argparse
import os

import misc_data_io as misc
import ch_util.andata
import ch_util.ephemeris as eph
import ch_pulsar_analysis as chp

from mpi4py import MPI
comm = MPI.COMM_WORLD
print comm.rank, comm.size

chand = ch_util.andata.AnData()

autos = [7,7]
nfreq = []
ncorr = []

ant = [0,1,2,4,5,6]
pairs = [(0,13), (1,14), (2,15), (4,9), (5,10), (6,11)] # These are the antenna pairs for layout 29, I think.  


parser = argparse.ArgumentParser(description="")
parser.add_argument("data_file", help="Data file to analyze")
parser.add_argument("-nfreq", help="number of frequencies", default=1024, type=int)
parser.add_argument("-ncorr", help="number of correlation products", type=int, default=16)
parser.add_argument("-freqwise", help="number of correlation products", default=1)
parser.add_argument("-skip_corr", default=0, type=int)
args = parser.parse_args()

nfreq = args.nfreq
ncorr = args.ncorr

rank = comm.rank

outfile = '/scratch/k/krs/connor/chime/calibration/20140519T192458Z/B0329+5420140519T192458Z1_corrmat.hdf5'

f = h5py.File(args.data_file + np.str(rank)+'.hdf5','r')
data = f['folded_arr'][:]

bin_tup = chp.find_ongate(psr_arr, ref_prod=[autos[0], autos[1]])
on_gate = arr_corr.shape[-1] / 2.
psr_vis = 0.80 * arr_corr[..., on_gate] + 0.10 * (arr_corr[..., on_gate+1] + arr_corr[..., on_gate-1]) -\
                0.5 * (arr_corr[..., on_gate+5:].mean(axis=-1) + arr_corr[..., :on_gate - 5].mean(axis=-1))


psr_vis_full = comm.gather(psr_vis, root=0)

if rank==0:
    psr_vis_full = np.concatenate(psr_vis_full, axis=0)
    for corr in range(ncorr):
        psr_vis_cal[:, corr] = misc.svd_model(psr_vis[:, corr])
        outfile = args.data_file + np.str(nfbin) + "psr_vis.hdf5"

        print "Writing to", outfile
        g = h5py.File(outfile, 'w')
        g.create_dataset("psr_visibilities", data = psr_vis_cal)
        g.close()

    ntimes = psr_vis_cal.shape[-1]

    v3 = np.zeros([nfreq, 16, ntimes], np.complex128)
    v7 = np.zeros([nfreq, 16, ntimes], np.complex128)
    v26 = np.zeros([2, 2, nfreq, ntimes], np.complex128)
    vx = np.zeros([2, 2, len(ant), nfreq, ntimes], np.complex128)

    v26[0,0] = v3[:, 3]
    v26[0,1] = v3[:, 7]
    v26[1,0] = v7[:, 3]
    v26[1,1] = v7[:, 7]

    print "Made 26m corrmat"

    for ii in range(len(ant)):
        print pairs[ii]
        vx[0,0, ii] = v3[:, pairs[ii][0]]
        vx[0,1, ii] = v3[:, pairs[ii][0]]
        vx[1,0, ii] = np.conj(v3[:, pairs[ii][0]])
        vx[1,1, ii] = v7[:, pairs[ii][0]]

    print "Writing to %s" % outfile
    f = h5py.File(outfile,'w')
    f.create_dataset('26m_corrmat',data=v26)
    f.create_dataset('x_corrmat', data=vx)
    f.close()





