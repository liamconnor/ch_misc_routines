import numpy as np

import ch_util.ephemeris as eph
import misc_data_io as misc
import h5py

def feed_map(feed_i, feed_j, nfeed):
     """
     Calculates correlation index between two feeds

     Parameters
     ==========
     feed_i, feed_j: int 
          Feed numbers counting from 0
     nfeed: int 
          Number of feeds (duhhh)

     Returns
     =======
     Correlation index
     """
     if feed_i > feed_j:
          return ((2*nfeed*feed_j - feed_j**2 + feed_j)/2) + (feed_i - feed_j)
     else:
          return ((2*nfeed*feed_i - feed_i**2 + feed_i)/2) + (feed_j - feed_i)

def gen_corr_matrix(data, nfeed, feed_loc=False):
     """
     Generates Hermitian (nfeed, nfeed) correlation matrix from unique correlations

     Parameters
     ==========
     data: (nfreq, ncorr, ntimes) np.complex128 arr
          Visibility array to be decomposed
     nfeed: int 
          Number of feeds (duhhh)

     Returns
     =======
     Hermitian correlation matrix
     """

     if not feed_loc:
          feed_loc=range(nfeed)
          
     corr_mat = np.zeros((len(feed_loc), len(feed_loc)), np.complex128)

     for ii in range(len(feed_loc)):
          for jj in range(ii, len(feed_loc)):
               corr_mat[ii, jj] = data[feed_map(feed_loc[ii], feed_loc[jj], nfeed)]
               corr_mat[jj, ii] = np.conj(data[feed_map(feed_loc[ii], feed_loc[jj], nfeed)])

     return corr_mat

def iterate_sol(data, nfeed):
     """
     Iteratively runs the eigendecomposition gain_sol_transit.solve_gain by
     using the previous solution to fill in autocorrelations.
     
     Parameters
     ==========
     data: (nfreq, ncorr, ntimes) np.complex128 arr
          Visibility array to be decomposed
     nfeed: int 
          Number of feeds in configuration

     Returns
     =======
     gain_arr: (nfreq, nfeed, ntimes) np.complex128
          Gain solution for each feed, time, and frequency
     eval_arr: (nfreq, ntimes)
          Eigenspectrum for each time and frequency
     """
     auto_ind = []
     [auto_ind.append(feed_map(i, i, nfeed)) for i in range(nfeed)]

     for ii in range(2):
          print "Iteration %d" % ii
          gain_arr, eval_arr = solve_gain(data, nfeed)
          data[:, auto_ind] = abs(gain_arr)**2

     return gain_arr, eval_arr

def solve_gain(data, nfeed, feed_loc=False):
     """
     Steps through each time/freq pixel, generates a Hermitian (nfeed,nfeed)
     matrix and calculates gains from its largest eigenvector.

     Parameters
     ==========
     data: (nfreq, ncorr, ntimes) np.complex128 arr
          Visibility array to be decomposed
     nfeed: int 
          Number of feeds in configuration

     Returns
     =======
     gain_arr: (nfreq, nfeed, ntimes) np.complex128
          Gain solution for each feed, time, and frequency
     eval_arr: (nfreq, ntimes)
          Eigenspectrum for each time and frequency
     """
     if not feed_loc:
          feed_loc=range(nfeed)     

     gain_arr = np.zeros([data.shape[0], len(feed_loc), data.shape[-1]], np.complex128)
     eval_arr = np.zeros([data.shape[0], len(feed_loc), data.shape[-1]], np.float64)

     for nu in range(data.shape[0]):
          if (nu%64)==0:
               print "Freq %d" % nu
          for tt in range(data.shape[-1]):
               corr_arr = gen_corr_matrix(data[nu, :, tt], nfeed, feed_loc=feed_loc)
               corr_arr[np.diag_indices(len(corr_arr))] = 0.0

               evl, evec = np.linalg.eigh(corr_arr) 
                                                           
               eval_arr[nu, :, tt] = evl
               gain_arr[nu, :, tt] = evl[-1]**0.5 * evec[:, -1]

     return gain_arr, eval_arr


if __name__ == '__main__':
     import argparse
     
     parser = argparse.ArgumentParser(description="Reads in point source transit and solve for complex gains")
     parser.add_argument("files", help=".h5 files to read")
     args = parser.parse_args()

     feed_locx = [0,1,2,12,13,14,15]
     feed_locy = [4,5,6,8,9,10,11]
     print args.files
     X, vis, t, RA, fpga_count = misc.get_data(args.files + '*')
     vis = vis #- np.median(vis[..., 175:450, np.newaxis], axis=-2)
     print "Read in data with shape:", vis.shape

     print "Starting xpol eigendecomposition"
     gainmatx, evmatx = solve_gain(np.median(vis.reshape(-1, 1, 136, vis.shape[-1]), 1), 16, feed_loc=feed_locx)
     print "Starting ypol eigendecomposition"
     gainmaty, evmaty = solve_gain(np.median(vis.reshape(-1, 1, 136, vis.shape[-1]), 1), 16, feed_loc=feed_locy)

     gainmatx *= gainmatx[:,0,np.newaxis]
     gainmaty *= gainmaty[:,0,np.newaxis]

     g = h5py.File('gainsolmed.hdf5','w')
     g.create_dataset('gainmatx', data=gainmatx)
     g.create_dataset('gainmaty', data=gainmaty)
     g.create_dataset('evmatx', data=evmatx)
     g.create_dataset('evmaty', data=evmaty)
     g.close()




