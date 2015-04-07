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

parser = argparse.ArgumentParser(description="")
parser.add_argument("data_file", help="Data file to analyze")
parser.add_argument("-outtag", help="", default="vis_full")
parser.add_argument("-nfreq", help="number of frequencies", default=1024, type=int)
parser.add_argument("-ln", help="Layout number", default='99')
args = parser.parse_args()

RA_src, dec_src = 53.51337, 54.6248916
ln = args.ln
nfreq = args.nfreq
nm = '/scratch/k/krs/connor/chime/calibration/' + args.data_file
outfile = nm[-18:-1]

rank = comm.rank
corr = rank

x26 = 0

psr_vis = []

mask = np.array([   0,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,
        124,  125,  126,  127,  128,  129,  130,  131,  132,  133,  134,
        135,  136,  137,  141,  142,  143,  144,  145,  146,  147,  148,
        149,  150,  151,  152,  153,  243,  256,  268,  273,  278,  384,
        485,  512,  521,  522,  553,  554,  555,  556,  557,  558,  559,
        560,  561,  562,  563,  564,  565,  566,  567,  568,  569,  570,
        571,  572,  573,  574,  575,  576,  577,  578,  579,  580,  581,
        582,  583,  584,  585,  586,  587,  588,  589,  590,  591,  592,
        593,  594,  595,  596,  597,  598,  599,  630,  631,  632,  633,
        634,  635,  636,  637,  638,  639,  640,  641,  642,  643,  644,
        645,  646,  647,  648,  649,  650,  651,  652,  653,  654,  655,
        656,  657,  658,  659,  660,  676,  677,  678,  679,  680,  681,
        682,  683,  684,  685,  686,  687,  688,  689,  690,  691,  752,
        753,  754,  755,  756,  757,  758,  759,  760,  761,  762,  763,
        764,  765,  766,  767,  768,  784,  785,  786,  787,  788,  789,
        790,  791,  792,  793,  794,  795,  796,  797,  798,  808,  809,
        810,  846,  848,  855,  856,  857,  858,  859,  866,  868,  873,
        875,  876,  878,  879,  882,  883,  889,  890,  893,  895,  920,
        928,  929,  930,  974,  975,  977,  978,  987,  988,  989,  990,
        991,  997, 1017, 1020, 1021])


Arr_chx = []
weights = []

print rank

for nu in range(nfreq):

    if nu % 128 == 0: 
        if rank == 0:
            print "Freq %d" % nu

    f = h5py.File(nm + np.str(nu) + '.hdf5', 'r')

    corrs = f.attrs['corrs']

    if rank==x26:
        fncount = f['folded_arr'][[x26], :, :].repeat(3, axis=0)
    else:
        fncount = f['folded_arr'][(x26, rank, rank+15), :, :]

    ic = f['icount'][:]
    
    arr = fncount / ic
    arr[np.isnan(arr)] = 0.0

    if nu in mask:
        arr *= 0.0

    del fncount, ic

    arr_26 = arr[[0]]
    arr_chx = arr[[1]]
    arr_chautos = arr[[2]]

    assert arr_chautos.sum().imag == 0.0
    assert arr_26.sum().imag == 0.0

    arr_26_tm = arr_26[np.newaxis].mean(-1)
    arr_chx_tm = arr_chautos.mean(-1)

    autos_gmean = np.sqrt(abs(arr_26_tm).mean(-1) *\
                     abs(arr_chx_tm).mean(-1))[:, np.newaxis, np.newaxis]
    
    arr_chx = arr_chx[np.newaxis] / autos_gmean
    arr_chx[np.isnan(arr_chx)] = 0.0

    Arr_chx.append(arr_chx)

    del autos_gmean, arr_26_tm, arr_chx_tm
    del arr_chx, arr_chautos

    weights.append(arr_26.mean(-2))

print arr_26.shape

times = f['times'][:]
times = np.linspace(times[0], times[-1], arr_26.shape[1])

Arr_chx = np.concatenate(Arr_chx, axis=0)[:, 0]
weights = np.concatenate(weights, axis=0)[:, np.newaxis]

psr_vis = chp.opt_subtraction(Arr_chx, weights=weights)

outfile = '/scratch/k/krs/connor/' + np.str(rank) + outfile + args.outtag + '.hdf5'
os.system('rm -f ' + outfile)

print outfile

print psr_vis.shape

RC = chp.PulsarPipeline(psr_vis[:, np.newaxis], times)
RC.nfreq = range(nfreq)
RC.RA_src, RC.dec = RA_src, dec_src        
RC.ln = ln      
RC.RA = eph.transit_RA(times)
RC.corrs = corrs[rank]
RC.fringestop(reverse=False, uf=0.885, vf=0.0)

print RC.RA
print RC.data.shape

# Assume beam is 3 degrees, normalized for declination
RA_max = RA_src + 2.5 / np.cos(np.radians(dec_src)) 
RA_min = RA_src - 2.5 / np.cos(np.radians(dec_src))
 
beam_ra = np.where((RC.RA < RA_max) & (RC.RA > RA_min))[0]

psr_vis = misc.correct_delay(RC.data[:, 0], nfreq=nfreq)

res_phase = np.angle(psr_vis[:, beam_ra].mean(-1).mean(0))
psr_vis *= np.exp(-1j*res_phase)
  
mask_time = misc.running_mean_rfi(abs(psr_vis.mean(0)), beam_ra=beam_ra, n=20)

g = h5py.File(outfile, 'w')
g.create_dataset("psr_vis", data=psr_vis)
g.create_dataset("time", data=times)

g.attrs.create("beam_ra", data=beam_ra)
g.attrs.create("mask_time", data=mask_time)

g.close()

