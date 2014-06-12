import numpy as np
from scipy.stats import mode
import h5py
import argparse

import misc_data_io as misc
import lag_correct as lag_cor

parser = argparse.ArgumentParser(description="This script locates the pulse bin, subtracts the offgates and calculates and corrects for lag")
parser.add_argument("data_file", help="Folded data array with shape (nfreq, ncorr, ntimes, nbins)")
args = parser.parse_args()


outdir = '/scratch/k/krs/connor/chime/calibration/'
dat_name = args.data_file[-35:-12]
dir_name = dat_name[:-7]

ncorr = 36

file = h5py.File(args.data_file)
data_allcorr = file['folded_arr'][:]
file.close()

on_gate = np.zeros((ncorr))

for corr in range(ncorr):
    profile = abs(data_allcorr[:, corr, :]).mean(axis=0).mean(axis=0)
    on_gate[corr] = np.where(profile==profile.max())[0][0]

mode_on = mode(on_gate)

print ""
print "%i/%i correlations peak in gate %i" % (mode_on[1], ncorr, mode_on[0])
print ""

PhAn = lag_cor.PhaseAnalysis(mode_on[0][0])

lag_pixel = PhAn.get_lag_pixel(data_allcorr)
lag_sol = PhAn.solve_lag(lag_pixel)

print "Delay in pixels:", np.round(lag_sol, 1)
print "Delays in meters:", np.round(lag_sol * 0.85 * 3e8 / 400e6, 1)
print ""

data_zerolag = PhAn.correct_lag()[0]

outfile = outdir + dir_name + '/' + dat_name + 'onegate.hdf5'
print "Writing to", outfile
file_out = h5py.File(outfile, 'w')

file_out.create_dataset('data_zerolag', data=data_zerolag)
file_out.create_dataset('data_onpulse', data=PhAn.on_pulse_data)
file_out.create_dataset('lag_sol', data=lag_sol)

file_out.close()

