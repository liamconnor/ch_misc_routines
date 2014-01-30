import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

DM = 26.833 # B0329 dispersion measure
p1 = 0.7145817552986237 # B0329 period

class PulsarPipeline:

    def __init__(self, data_arr, time_stamps, time_res=0.010):
        self.data = data_arr.copy()
        self.ntimes = self.data.shape[-1]
        self.time_stamps = time_stamps
        
        if self.ntimes != len(self.time_stamps):
            raise Exception('Number of samples disagree')
            
        self.time_res = time_res
        self.data = np.ma.array(self.data, mask=np.zeros(self.data.shape, dtype=bool))
        self.nfreq = self.data.shape[0]
        self.highfreq = 800.0
        self.lowfreq = 400.0
        self.freq = np.linspace(self.highfreq, self.lowfreq, self.nfreq)

        self.ntimes = self.data.shape[-1]
        print "Data array has shape:", self.data.shape

    def dm_delays(self, dm, f_ref):
        """
        Provides dispersion delays as a function of frequency. 
        
        Parameters
        ----------
        dm : float
                Dispersion measure in pc/cm**3
        f_ref : float
                reference frequency for time delays
                
        Returns
        -------
        Vector of delays in seconds as a function of frequency
        """
        return 4.148808e3 * dm * (self.freq**(-2) - f_ref**(-2))
    
    def fold_pulsar(self, p0, dm, nbins=32, **kwargs):
        """
        Folds pulsar into nbins time bins after dedispersing it. 
        
        Parameters
        ----------
        p0 : float
                Pulsar period in seconds. 
        dm : float
                Dispersion measure in pc/cm**3
                
        Returns
        -------
        profile: complex vec 
                Folded pulse profile of length nbins

        """        
        if kwargs.has_key('start_chan'): start_chan = kwargs['start_chan']
        else: start_chan = 0
        if kwargs.has_key('end_chan'): end_chan = kwargs['end_chan']
        else: end_chan = self.nfreq
        if kwargs.has_key('start_samp'): start_samp = kwargs['start_samp']
        else: start_samp = 0
        if kwargs.has_key('end_samp'): end_samp = kwargs['end_samp']
        else: end_samp = self.ntimes   
            
        times = self.time_stamps[start_samp:end_samp]
        freq = self.freq[start_chan:end_chan]
        if kwargs.has_key('f_ref'): f_ref = kwargs['f_ref']
        else: f_ref = freq[0]
   
        data = self.data[start_chan:end_chan, start_samp:end_samp].copy()
        data /= running_mean(data)
        delays = self.dm_delays(dm, f_ref)[start_chan:end_chan].repeat(len(times)).reshape(data.shape)
        dedispersed_times = times[np.newaxis, :] * np.ones([len(freq), 1]) - delays
        
        bins = (((dedispersed_times / p1) % 1) * nbins).astype(int)
        profile = np.zeros([nbins], dtype=np.complex128)
        
        for i in range(nbins):
            vals = data[bins==i]
            profile[i] += vals[vals!=0.0].mean()
        
        return profile
        
        #profile2d = np.zeros([len(freq), nbins], dtype=np.complex128)
        #for bin in range(nbins):
        #    for nu in range(len(freq)):
        #        vals = data[nu, bins[nu]==bin]
        #        profile2d[nu, bin] += vals[vals!=0.0].mean()

        #return profile2d
    
class RFI_Clean(PulsarPipeline):

    always_cut = range(111,138) + range(889,893) + range(856,860) + \
        range(873,877) + range(583,600) + range(552,568) + range(630,645) +\
        range(678,690) + range(753, 768) + range(783, 798)
    
    def frequency_clean(self, threshold=1):
        """
        Does an RFI cut in frequency by normalizing by nu**4 to flatten bandpass and then cutting 
        above some threshold. 
        
        Parameters
        ----------
        data : array of np.complex128
                pulsar data array. 
        threshold :
                threshold at which to cut spectral RFI. For autocorrelations threshold=1 is reasonable, for
                cross-correlations 0.25 seems to work.
                
        Returns
        -------
        RFI-cleaned data

        """
        data = abs(self.data.copy())
        data_freq = data.mean(axis=-1)
        data_freq_norm = data_freq / data_freq.mean() * self.freq**(4) / self.freq[-1]**4
        mask = data_freq_norm > threshold

        self.data[np.where(mask)[0], :] = 0.0 
        self.data[self.always_cut, :] = 0.0
        print "Finished cutting spectral RFI"
        
    def time_clean(self, threshold=5):
        """
        Does an RFI cut in time by normalizing my nu**4 to flatten bandpass and then cutting 
        above some threshold. SHOULD NOT USE THIS AT THE MOMENT, IT INTERFERES WITH FOLDING
        
        Parameters
        ----------
        data : array of np.complex128
                pulsar data array. 
        threshold :
                threshold at which to cut transient RFI. Threshold of 5 seems reasonable.
                
        Returns
        -------
        RFI-cleaned data

        """
        data = self.data
        data_time = data.mean(axis=0)
        data_time_norm = abs(data_time - np.median(data_time))
        data_time_norm /= np.median(data_time_norm)
        mask = data_time_norm > threshold
        self.data[:, np.where(mask)[0]] = 0.0 
        print "Finished cutting temporal RFI"

def running_mean(arr, radius=50):
    """                                                                                                                
    Not clear why this works. Need to think more about it.                                                                             
    """
    n = radius*2+1
    padded = np.concatenate((arr[:, 1:radius+1][:, ::-1], arr,\
        arr[:, -radius-1:-1][:, ::-1]), axis=1)
    ret = np.cumsum(padded, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    
    return ret[:, n-1:] / n


