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

autos = [45, 91]

parser = argparse.ArgumentParser(description="")
parser.add_argument("data_file", help="Data file to analyze")
parser.add_argument("-outtag", help="", default="vis_full")
parser.add_argument("-nfreq", help="number of frequencies", default=1024, type=int)
parser.add_argument("-svd", help="Remove largest phase eigenmode with SVD", type=int, default=1)
parser.add_argument("-ln", help="Layout number", default='29')
parser.add_argument("-use_chime_autos", help="only chime autos were folded", default=0)
args = parser.parse_args()

nfreq = args.nfreq

rank = comm.rank

psr_arr = []

psr_arrx = []
psr_arr26 = []
psr_auto_i = []
psr_auto_j = []

nfeed = 16
corrs = range(nfeed*(nfeed+1)/2)

autos_i = []
autos_j = []

for i in range(nfeed):
    for j in range(i, nfeed):
        autos_i.append(misc.feed_map(i, i, 16))
        autos_j.append(misc.feed_map(j, j, 16))

for nu in range(nfreq):

    if nu % 128 == 0: 
        print "Freq %d" % nu

    f = h5py.File(args.data_file + np.str(nu) + '.hdf5', 'r')

    datax = f['folded_arr'][corrs[rank], 2000:9000]
    data_auto_i = f['folded_arr'][[autos_i[rank]], 2000:9000]
    data_auto_j = f['folded_arr'][[autos_j[rank]], 2000:9000]
    data26m = f['folded_arr'][autos, 2000:9000]

    psr_arrx.append(datax[np.newaxis])
    psr_auto_i.append(data_auto_i)
    psr_auto_j.append(data_auto_j)
    psr_arr26.append(data26m[np.newaxis])
    
t = f['times']
ntimes = f['folded_arr'][:].shape[1]

print "ntimes", ntimes

time = np.linspace(t[0], t[-1], ntimes)
time = time[2000:9000]

psr_arrx = np.concatenate(psr_arrx)
psr_auto_i = np.concatenate(psr_auto_i)
psr_auto_j = np.concatenate(psr_auto_j)
psr_arr26 = np.concatenate(psr_arr26)

print "Done concatenating frequency files"

psr_arrx[np.isnan(psr_arrx)] = 0.0
psr_auto_i[np.isnan(psr_auto_i)] = 0.0
psr_auto_j[np.isnan(psr_auto_j)] = 0.0
psr_arr26[np.isnan(psr_arr26)] = 0.0

print psr_auto_i.shape, psr_auto_j.shape

psr_auto_i = abs(psr_auto_i).mean(-1).mean(-1)[:, np.newaxis, np.newaxis]
psr_auto_j = abs(psr_auto_j).mean(-1).mean(-1)[:, np.newaxis, np.newaxis]

psr_arrx /= (np.sqrt(psr_auto_i * psr_auto_j) + 1e-18)

del psr_auto_i, psr_auto_j

ntimes = psr_arrx.shape[-2]

#bin_tup = chp.find_ongate(abs(psr_arr26), ref_prod=[0,1])
#arr_corr = chp.correct_phase_bins(psr_arrx, bin_tup)
#arr_corr26 = chp.correct_phase_bins(psr_arr26, bin_tup)

arr_corr = psr_arrx
arr_corr26 = psr_arr26

on_gate = arr_corr.shape[-1] / 2.
on_gate = 13

if on_gate < 15:
    on_vis = arr_corr[..., on_gate]
#    off_vis = 0.5 * (arr_corr[..., on_gate-2].mean(-1) + arr_corr[..., on_gate+2:].mean(-1))
    off_vis = 0.5 * (arr_corr[..., on_gate+2] \
                          + arr_corr[..., on_gate-5]) # For only 14 gates
elif on_gate > 9:
    on_vis = 0.80 * arr_corr[..., on_gate] + \
        0.10 * (arr_corr[..., on_gate+1] + arr_corr[..., on_gate-1])
    off_vis = 0.5 * (arr_corr[..., on_gate-3] + \
        arr_corr[..., on_gate+3])
    
psr_vis = on_vis - off_vis
 
x = 7
y = 3

print psr_vis.shape, time.shape
RC = chp.PulsarPipeline(psr_vis[:, np.newaxis], time)
RC.RA_src, RC.dec = 53.51337, 54.6248916
RC.ln = args.ln
RC.RA = eph.transit_RA(time)
RC.corrs = corrs[rank]
RC.fringestop()

psr_vis = RC.data[:, 0]

print "0"
psr_vis = misc.correct_delay(psr_vis, nfreq=nfreq)
print "1"

if args.svd == 1:
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



