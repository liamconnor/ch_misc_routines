import numpy as np
from numpy import linalg
import h5py

n_ant = 8
nfreq = 1024
ncorr = n_ant * (n_ant+1) / 2
kl_max = 1

g = h5py.File('/scratch/k/krs/connor/B0329_10dec13_ongate.hdf5','r')
onegate_ = g['data'][:] # Should have shape (nfreq, ncorr, ntimes)
print onegate_.shape

onegate = np.transpose(onegate_, (1,2,0)) # Should be ordered (ncorr, ntime, nfreq)                                        
print "";print "Opening onegate.float:",onegate.shape;print ""

ntimes = onegate.shape[1]

print 'ntimes', ntimes

def cmedian(arr, ax):
	"""
	Gets complex median in a way that mimics exactly Ue-Li's version.
	"""
	arr_med = arr.shape[ax] * (np.median(arr.real,axis=ax) + 1.0J*np.median(arr.imag,axis=ax))
	 
	return arr_med

def antmap(i,j,n):
    i=i-1;j=j-1
    if i>j:
        return ((2*n*j - j**2 + j) / 2) + (i-j)
    else:
        return ((2*n*i - i**2 + i) / 2) + (j-i)

def get_eigenvectors(Data, N):
    """ Function to get eigenvectors in XX and YY for each time or frequency
    
    ====params====
    Data:
    	File with pulsar visibilites                                                                                                    
    N:
    	Either number of frequencies or number of times on which to solve e-vec                                                            
    """
    Gain_matXX = np.zeros([n_ant, n_ant, N], np.complex64)

    for ant_i in range(n_ant):
    	for ant_j in range(n_ant):
    		if ant_i < ant_j:
    			Gain_matXX[ant_i,ant_j,:] = Data[antmap(ant_i+1, ant_j+1, n_ant), :]
    			Gain_matXX[ant_j,ant_i,:] = np.conj(Data[antmap(ant_i+1, ant_j+1, n_ant), :]) #Should be Hermitian 

                Gain_matXX[ant_j,ant_j,:] = 0 # Set autocorrelations to zero                                          

    # Now try to find eigenvectors of n_ant x n_ant gain matrices                                                                      
    gain_approx_XX = np.zeros([n_ant, N], np.complex64)

    for k in range(N):
    	w_XX, v_XX = np.linalg.eigh(Gain_matXX[:, :, k], UPLO='U')
    	v_XX = v_XX[:, -1] / v_XX[0, -1]
    	gain_approx_XX[:, k] = w_XX[-1]**0.5 * v_XX / np.sqrt(abs(v_XX**2).sum())

    return gain_approx_XX 


def unique_corr(g,n):
	"""
	Function to get n_ant*(n_ant+1)/2 unique correlations
	from total n_ant**2 correlations
	"""
	print g.shape
	corr_all = g[:,np.newaxis,:] * np.conj(g[np.newaxis,:,:])
	corr_unique = np.zeros([ncorr, n], np.complex64)

	k=0
	for ant_i in range(n_ant):
		for ant_j in range(n_ant):
			if ant_i <= ant_j:
				corr_unique[k,:] = corr_all[ant_i, ant_j,:]
				k=k+1

	return corr_unique

onegate_time = cmedian(onegate, 1) # Take time cmedian to get bandpass calibration         

g_freq = get_eigenvectors(onegate_time, nfreq)
g_freq[ g_freq!=g_freq ] = 0.0
#A = get_eigenvectors(onegate_time, nfreq)

print "%f NaNs" % np.isnan(g_freq).sum()
print 'Solving freq eigenvectors'

G_freq = unique_corr(g_freq, nfreq)

for kl in range(kl_max):
	print ""
	print "kl=",kl	
	
	onegate_cal = onegate * np.conj(G_freq[:,np.newaxis,:]) / abs(G_freq[:,np.newaxis,:] + 1e-16) / (abs(G_freq + 1e-16)/nfreq/ncorr).sum()
	onegate_freq = cmedian(onegate_cal, -1)
	print "%f NaNs" % np.isnan(onegate_cal).sum()

	print 'Solving time eigenvectors'

	g_time = get_eigenvectors(onegate_freq, ntimes)

	G_time = unique_corr(g_time, ntimes)

	G_time = G_time * (np.conj(G_time[:,0]) / abs(G_time[:,0]))[:, np.newaxis] # Set first timestamp to zero phase
	G_time = G_time / np.sqrt((abs(G_time**2)).sum(axis=1)/ntimes)[:, np.newaxis]  # Set average gain in time to 1 for each antenna
	onegate_time_cal = onegate * np.conj(G_time[:,:,np.newaxis])

	onegate_time = cmedian(onegate_time_cal,1)
	g_freq = get_eigenvectors(onegate_time, nfreq)
	
	G_freq = unique_corr(g_freq, nfreq) # Equivalent to cmatt in pulsar_llrr.f90

print "Done with Eigenstuff!"

gain = g_time[:,np.newaxis,:] * np.conj(g_freq[:,:,np.newaxis])

model = G_freq[:, np.newaxis, :] * G_time[:, :, np.newaxis]
model = model.reshape(ncorr, nfreq, ntimes)

print "getting model"

modelraw = np.transpose(model,(2,0,1))
datacal = onegate * np.conj(G_freq)[:,np.newaxis,:] * np.conj(G_time)[:,:,np.newaxis]
modelcal = (G_freq * np.conj(G_freq))[np.newaxis,:,:] * abs(np.transpose(G_time)[:,:,np.newaxis])**2 / abs(np.transpose(G_time)[:,:,np.newaxis]**2).sum(axis=0)

datacal = np.transpose(datacal,(1,0,2))

print "writing to file"
f = h5py.File('/Users/liamconnor/Desktop/calibration_data.hdf5','w')
f.create_dataset('datacal', data=datacal)
f.create_dataset('modelraw', data=modelraw)
f.create_dataset('modelcal', data=modelcal)
f.close()
