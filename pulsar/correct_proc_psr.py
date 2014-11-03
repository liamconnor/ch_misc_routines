import os

import numpy as np
import h5py
import argparse

import misc_data_io as misc
import ch_util.andata
import ch_util.ephemeris as eph
import ch_pulsar_analysis as chp

from mpi4py import MPI
comm = MPI.COMM_WORLD
print comm.rank, comm.size

chand = ch_util.andata.AnData()

autos = [7, 19]

ant = [0,1,2,4,5,6]
pairs = [(0,13), (1,14), (2,15), (4,9), (5,10), (6,11)] # These are the antenna pairs for layout 29, I think.  

parser = argparse.ArgumentParser(description="")
parser.add_argument("data_file", help="Data file to analyze")
parser.add_argument("-outtag", help="", default="vis_full")
parser.add_argument("-nfreq", help="number of frequencies", default=1024, type=int)
parser.add_argument("-svd", help="Remove largest phase eigenvalue with SVD", type=int, default=1)
parser.add_argument("-ln", help="Layout number", default='29')
args = parser.parse_args()

nfreq = args.nfreq

rank = comm.rank 

psr_arr = []

psr_arrx = []
psr_arr26 = []

for nu in range(nfreq):

    if nu % 128 ==0: print "Freq %d" % nu

    f = h5py.File(args.data_file + np.str(nu) + '.hdf5','r')
    datax = f['folded_arr'][[rank]]
    data26m = f['folded_arr'][autos]
    psr_arrx.append(datax[np.newaxis])
    psr_arr26.append(data26m[np.newaxis])
    
print "Done concatenating frequency files"

t = f['times']
ntimes = f['folded_arr'][:].shape[1]
print ntimes
time = np.linspace(t[0], t[-1], ntimes)
time = time[:]

psr_arrx = np.concatenate(psr_arrx)
psr_arr26 = np.concatenate(psr_arr26)

psr_arrx[np.isnan(psr_arrx)] = 0.0
psr_arr26[np.isnan(psr_arr26)] = 0.0

ntimes = psr_arrx.shape[-2]

bin_tup = chp.find_ongate(abs(psr_arr26), ref_prod=[0,1])
arr_corr = chp.correct_phase_bins(psr_arrx, bin_tup)
arr_corr26 = chp.correct_phase_bins(psr_arr26, bin_tup)

on_gate = arr_corr.shape[-1] / 2.


if on_gate<10:
    psr_vis = arr_corr[..., on_gate] - 0.5 * ( arr_corr[..., on_gate+2] + arr_corr[..., on_gate-2] ) # For only 14 gates
elif on_gate>9:
    psr_vis = 0.80 * arr_corr[..., on_gate] + 0.10 * (arr_corr[..., on_gate+1] + arr_corr[..., on_gate-1]) \
        - 0.5 * (arr_corr[..., on_gate-3] + arr_corr[..., on_gate+3])
    
x=7
y=3

nfeeds=16
feeds = np.arange(nfeeds)
corrs = [misc.feed_map(i, x, nfeeds) for i in feeds] + [misc.feed_map(i, y, nfeeds) for i in feeds]
 
RC = chp.PulsarPipeline(psr_vis, time)
RC.RA_src, RC.dec = 53.51337, 54.6248916
RC.ln = args.ln
RC.RA = eph.transit_RA(time)
RC.corrs = corrs[rank]
RC.fringestop()

psr_vis = RC.data[:,0]

psr_vis = misc.correct_delay(psr_vis, nfreq=nfreq)

if args.svd==1:
    psr_vis_cal = misc.svd_model(psr_vis, phase_only=True)
    print "Performing SVD on dynamic spectrum, rank %d" % rank
else:
    print "Skipping SVD"
    psr_vis_cal = psr_vis[:]

outfile = args.data_file + np.str(rank) + args.outtag + ".hdf5"
print "Writing to", outfile
print "with shape:", psr_vis_cal.shape

os.system('rm -f ' + outfile)

g = h5py.File(outfile, 'w')
g.create_dataset("psr_vis", data=psr_vis_cal)
g.create_dataset("arr", data=psr_arrx)
g.create_dataset("time", data=time)
g.close()


"""
print psr_vis_cal.shape,"yo"

if rank==0:
    psr_vis = np.concatenate(psr_vis_full, axis=0)
    
    for corr in range(ncorr):
        psr_vis_cal[:, corr] = misc.svd_model(psr_vis[:, corr])
        

    outfile = args.data_file + "psr_vismf.hdf5"
    print "Writing to", outfile
    print "with shape:", psr_vis_cal.shape
    g = h5py.File(outfile, 'w')
    g.create_dataset("psr_visibilities", data = psr_vis)
    g.close()

    v3 = psr_vis_cal#np.zeros([nfreq, 16, ntimes], np.complex128)
    v7 = psr_vis_cal#np.zeros([nfreq, 16, ntimes], np.complex128)
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
    #f = h5py.File(outfile,'a')
    g.create_dataset('26m_corrmat',data=v26)
    g.create_dataset('x_corrmat', data=vx)
    g.close()
"""



