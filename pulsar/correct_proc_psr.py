import numpy as np
import h5py
import argparse
import os

import misc_data_io as misc
import ch_util.andata
import ch_util.ephemeris as eph
import ch_pulsar_analysis as chp
chand = ch_util.andata.AnData()

autos = [7,7]
nfreq = []
ncorr = []


parser = argparse.ArgumentParser(description="")
parser.add_argument("data_file", help="Data file to analyze")
parser.add_argument("-nfreq", help="number of frequencies", default=1024, type=int)
parser.add_argument("-ncorr", help="number of correlation products", type=int, default=16) 
parser.add_argument("-freqwise", help="number of correlation products", default=1)
parser.add_argument("-skip_corr", default=0, type=int)
args = parser.parse_args()

nfreq = args.nfreq
ncorr = args.ncorr

print args.skip_corr

#if os.path.exists('/scratch/k/krs/connor/chime/calibration/20140519T192458Z/B0329+5420140519T192458Z126m30psr_vis.hdf5'):
#    pass
if args.skip_corr==1:
    print "Skipping the big stuff"
    pass
else:
    if args.freqwise:
        psr_arr = []
        for nfbin in range(4):
            psr_arr_4 = []
            for nu in range(nfbin*nfreq/4, (nfbin+1)*nfreq/4):
                print nu
                f = h5py.File(args.data_file + np.str(nu)+'.hdf5','r')
                data = f['folded_arr'][:]
                psr_arr_4.append(data[np.newaxis])
            print "Read in", nfreq / 4, "files, full array shape:", len(psr_arr)
            psr_arr = np.concatenate(psr_arr_4)
            print nfbin
            print psr_arr.shape

            bin_tup = chp.find_ongate(psr_arr, ref_prod=[autos[0], autos[1]])
            arr_corr = chp.correct_phase_bins(psr_arr, bin_tup)

            print "Done correcting phase bins"

            on_gate = arr_corr.shape[-1] / 2.
            psr_vis = 0.80 * arr_corr[..., on_gate] + 0.10 * (arr_corr[..., on_gate+1] + arr_corr[..., on_gate-1]) -\
                0.5 * (arr_corr[..., on_gate+5:].mean(axis=-1) + arr_corr[..., :on_gate - 5].mean(axis=-1))

            psr_vis_cal = np.zeros_like(psr_vis)

            for corr in range(ncorr):
                psr_vis_cal[:, corr] = misc.svd_model(psr_vis[:, corr])

                outfile = args.data_file + np.str(nfbin) + "psr_vis.hdf5"

                print "Writing to", outfile
                g = h5py.File(outfile, 'w')
                g.create_dataset("psr_visibilities", data = psr_vis_cal)
                g.close()
        
            ntimes = psr_vis_cal.shape[-1]
    else:
        f = h5py.File(args.data_file,'r')
        psr_arr = f['folded_arr'][:]
        psr_t = f['times'][:]
        ntimes = psr_arr.shape[-1]

ntimes=1554
v3 = np.zeros([1024, 16, ntimes], np.complex128)
v7 = np.zeros([1024, 16, ntimes], np.complex128)
v26 = np.zeros([2,2, 1024, ntimes], np.complex128)
vx = np.zeros([2,2, 6, 1024, ntimes], np.complex128)

ant = [0,1,2,4,5,6]
pairs = [(0,13), (1,14), (2,15), (4,9), (5,10), (6,11)] # These are the antenna pairs for layout 29, I think.

for nfbin in range(4):
    print nfbin
    g = h5py.File('/scratch/k/krs/connor/chime/calibration/20140519T192458Z/B0329+5420140519T192458Z126m3' + np.str(nfbin) + 'psr_vis.hdf5','r')
    h = h5py.File('/scratch/k/krs/connor/chime/calibration/20140519T192458Z/B0329+5420140519T192458Z126m7' + np.str(nfbin) + 'psr_vis.hdf5','r')
    
    v3[256*nfbin:256*(1+nfbin)] = g['psr_visibilities'][:]
    v7[256*nfbin:256*(1+nfbin)] = h['psr_visibilities'][:]

    v26[0,0, 256*nfbin:256*(1+nfbin)] = g['psr_visibilities'][:][:, 3] # Define 26m channel 3 as "x" pol
    v26[0,1, 256*nfbin:256*(1+nfbin)] = g['psr_visibilities'][:][:, 7]
    v26[1,0, 256*nfbin:256*(1+nfbin)] = np.conj(g['psr_visibilities'][:][:, 7])
    v26[1,1, 256*nfbin:256*(1+nfbin)] = h['psr_visibilities'][:][:, 7]
    print "Made 26m corrmat"

    for ii in range(6):
        print pairs[ii]
        vx[0,0, ii, 256*nfbin:256*(1+nfbin)] = g['psr_visibilities'][:][:, pairs[ii][0]]
        vx[0,1, ii, 256*nfbin:256*(1+nfbin)] = h['psr_visibilities'][:][:, pairs[ii][0]]
        vx[1,0, ii, 256*nfbin:256*(1+nfbin)] = g['psr_visibilities'][:][:, pairs[ii][1]]
        vx[1,1, ii, 256*nfbin:256*(1+nfbin)] = h['psr_visibilities'][:][:, pairs[ii][1]]

    print "Writing to %s" % '/scratch/k/krs/connor/chime/calibration/20140519T192458Z/B0329+5420140519T192458Z1_corrmat.hdf5'
    f = h5py.File('/scratch/k/krs/connor/chime/calibration/20140519T192458Z/B0329+5420140519T192458Z1_corrmat.hdf5','w')
    f.create_dataset('26m_corrmat',data=v26)
    f.create_dataset('x_corrmat', data=vx)
    f.close()
    
