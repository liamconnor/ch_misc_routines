import numpy as np
import os
import h5py
import misc_data_io as misc

chime_lat = 49.320

class PulsarPipeline:

    def __init__(self, data_arr, time_stamps, time_res=0.010):
        self.data = data_arr.copy()
        self.ntimes = self.data.shape[-1]
        self.time_stamps = time_stamps

        if self.ntimes != len(self.time_stamps):
            print self.ntimes, "not equal to", len(self.time_stamps)
            raise Exception('Number of samples disagree')         
        
        self.RA = None
        self.dec = None
        self.RA_src = None

        self.time_res = time_res
        self.data = np.ma.array(self.data, mask=np.zeros(self.data.shape, dtype=bool))
        self.nfreq = self.data.shape[0]
        self.highfreq = 800.0
        self.lowfreq = 400.0
        self.freq = np.linspace(self.highfreq, self.lowfreq, self.nfreq)

        self.ntimes = self.data.shape[-1]
        self.ncorr = self.data.shape[1]
        self.corrs = range(self.ncorr)
        self.xcorrs = None
        self.ln = '29'
        self.powlaw = 0
        print "Data array has shape:", self.data.shape
        
    
    def get_uv(self):
        fname = '/home/k/krs/connor/code/ch_misc_routines/pulsar/feed_loc_layout' + self.ln + '.txt'
        feed_loc = np.loadtxt(fname)
        d_EW, d_NS = misc.calc_baseline(feed_loc)[:2]

        u = d_EW[np.newaxis, self.corrs, np.newaxis] \
            * self.freq[:, np.newaxis, np.newaxis] * 1e6 / (3e8)

        v = d_NS[np.newaxis, self.corrs, np.newaxis] \
            * self.freq[:, np.newaxis, np.newaxis] * 1e6 / (3e8)
        
        return u, v

    def dm_delays(self, dm, f_ref):
        """
        Provides dispersion delays as a function of frequency. Note
        frequencies are in MHz.
        
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


    def fold(self, dm, p0, ntrebin=100, ngate=32):
        """Folds pulsar into nbins after dedispersing it. 
        
        Parameters
        ----------
        p0 : float
                Pulsar period in seconds. 
        dm : float
                Dispersion measure in pc/cm**3
        ntrebin : np.int
                Number of time stamps that go into folded period
        ngate : np.int
                Number of phase bins to fold on
                
        Returns
        -------
        fold_arr : array_like 
                Folded complex array shaped (nfreq, ncorr, ntimes/ntrebin, ngate)
        icount : array_like
                Number of elements in given phase bin (nfreq, ntimes/ntrebin, ngate)

        """        

        ntimes = self.data.shape[-1] // ntrebin
        ncorr = self.data.shape[1]

        fold_arr = np.zeros([self.nfreq, ncorr, ntimes, ngate], np.complex128)
        icount = np.zeros([self.nfreq, ntimes, ngate], np.int32)

        dshape = self.data.shape[:-1] + (ntimes, ntrebin)

        data_rb = self.data[..., :(ntimes*ntrebin)].reshape(dshape)
        
        bins_all = self.phase_bins(p0, dm, ngate)

        for fi in range(self.nfreq):
            if (fi % 128) == 0:
                print "Folded freq", fi
            tau = self.dm_delays(dm, 400)

            bins = bins_all[fi, :(ntrebin*ntimes)].reshape(ntimes, ntrebin)

            for ti in range(ntimes):

                icount[fi, ti] = np.bincount(bins[ti], data_rb[fi, 0, ti] != 0., ngate)

                for corr in range(ncorr):

                    data_fold_r = np.bincount(bins[ti], 
                                              weights=data_rb[fi, corr, ti].real, minlength=ngate)
                    data_fold_i = np.bincount(bins[ti], 
                                              weights=data_rb[fi, corr, ti].imag, minlength=ngate)

                    fold_arr[fi, corr, ti, :] = data_fold_r + 1j * data_fold_i

        return fold_arr, icount[:, np.newaxis]

    def phase_bins(self, p0, dm, ngate):

        bins = np.zeros([self.nfreq, self.ntimes], np.int)
        
        times = self.time_stamps

        for fi in range(self.nfreq):

            tau = self.dm_delays(dm, 400)
            times_del = times - tau[fi]

            bins[fi] = (((times_del / p0) % 1) * ngate).astype(np.int)

        return bins

    def fold_two_periods(self, data, times, dm, p0_psr, p0_ns, 
                         ntrebin=100, ntrebin_ns=100, ngate=32, ngate_ns=32):
        """ Fold on both the noise source and the pulsar.
        """
        ntimes = data.shape[-1] // ntrebin
        ntimes_ns = data.shape[-1] // ntrebin_ns

        ncorr = data.shape[1]

        fold_arr = np.zeros([self.nfreq, ncorr, ntimes, ngate], np.complex128)
        icount = np.zeros([self.nfreq, ntimes, ngate], np.int32)

        fold_arr_ns = np.zeros([self.nfreq, ncorr, ntimes_ns, ngate_ns], np.complex128)
        icount_ns = np.zeros([self.nfreq, ntimes_ns, ngate_ns], np.int32)        

        dshape = data.shape[:-1] + (ntimes, ntrebin)
        dshape_ns = data.shape[:-1] + (ntimes_ns, ntrebin_ns)

        data_rb = data[..., :(ntimes*ntrebin)].reshape(dshape)
        data_rb_ns = data[..., :(ntimes_ns*ntrebin_ns)].reshape(dshape_ns)

        for fi in range(self.nfreq):

            if (fi % 128) == 0:
                print "Folded freq", fi

            tau = self.dm_delays(dm, 400)
            times_del = times - tau[fi]

            bins = (((times_del / p0_psr) % 1) * ngate).astype(np.int)
            bins = bins[:(ntrebin*ntimes)].reshape(ntimes, ntrebin)

            bins_ns = (((times / p0_ns) % 1) * ngate_ns).astype(np.int)
            bins_ns = bins_ns[:(ntrebin_ns*ntimes_ns)].reshape(ntimes_ns, ntrebin_ns)

            for ti in range(ntimes):

                icount[fi, ti] = np.bincount(bins[ti], data_rb[fi, 0, ti] != 0., ngate)

                for corr in range(ncorr):

                    data_fold_r = np.bincount(bins[ti], 
                                              weights=data_rb[fi, corr, ti].real, minlength=ngate)
                    data_fold_i = np.bincount(bins[ti], 
                                              weights=data_rb[fi, corr, ti].imag, minlength=ngate)

                    fold_arr[fi, corr, ti, :] = data_fold_r + 1j * data_fold_i

            for tj in range(ntimes_ns):

                icount_ns[fi, tj] = np.bincount(bins_ns[tj], data_rb_ns[fi, 0, tj] != 0., ngate_ns)

                for corr in range(ncorr):

                    data_fold_r = np.bincount(bins_ns[tj], 
                                              weights=data_rb_ns[fi, corr, tj].real, minlength=ngate_ns)
                    data_fold_i = np.bincount(bins_ns[tj], 
                                              weights=data_rb_ns[fi, corr, tj].imag, minlength=ngate_ns)

                    fold_arr_ns[fi, corr, tj, :] = data_fold_r + 1j * data_fold_i


        return fold_arr, icount[:, np.newaxis], fold_arr_ns, icount_ns[:, np.newaxis]

    
    def fold_pulsar(self, p0, dm, nbins=32, **kwargs):
        """
        Folds pulsar into nbins after dedispersing it. 
        
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
   
        data = self.data[start_chan:end_chan, :, start_samp:end_samp].copy()

        if self.powlaw==1:
            print "moving powlaw div"
            for corr in range(self.ncorr):
#            data[:, corr, :] /= running_mean(data[:, corr, :])
                data[:, corr, :] /= (abs(data[:,corr]).mean(axis=-1)[:, np.newaxis] / freq[:, np.newaxis]**(-0.8))

        delays = self.dm_delays(dm, f_ref)[start_chan:end_chan, np.newaxis, np.newaxis] * np.ones([1, data.shape[1], data.shape[-1]])
        dedispersed_times = times[np.newaxis, np.newaxis, :] * np.ones([data.shape[0], data.shape[1], 1]) - delays

        bins = (((dedispersed_times / p0) % 1) * nbins).astype(int)
        profile = np.zeros([data.shape[1], nbins], dtype=np.complex128)
        
        for corr in range(self.ncorr):
            for i in range(nbins):
                data_corr = data[:, corr, :]
                bins_corr = bins[:, corr, :]
                vals = data_corr[bins_corr==i]
                profile[corr, i] += vals[vals!=0.0].mean()
        
        return profile

    def noisefunc(self, data):
        ngate = data.shape[-1]

        on = [19, 20, 21]
        off = np.delete(range(ngate), on)
        
        data[np.isnan(data)] = 0.0
        profile_avg = data.mean(axis=1)
        on_avg = profile_avg[..., on] - profile_avg[..., np.newaxis, off].mean(-1)
        
        data_no_ns = data.copy()

        #Now recover the data without any noise injection
        data_no_ns[..., on] -= on_avg[:, np.newaxis]

        data_on = data[..., on] - np.median(data[..., np.newaxis, off], axis=-1)

        return data_no_ns, data_on
                        

    def fringestop(self, reverse=False):
        data = self.data.copy()

        ha = np.deg2rad(self.RA[np.newaxis, np.newaxis, :] - self.RA_src)
        dec = np.deg2rad(self.dec)

        u, v = self.get_uv()
        phase = self.fringestop_phase(ha, np.deg2rad(chime_lat), dec, 0.93*u, 0.93*v)
        
        if reverse==True:
            self.data = data * np.conj(phase)
            print "Reverse fringestopping"
        else:
            self.data = data * phase


    def fringestop_phase(self, ha, lat, dec, u, v):
        """Return the phase required to fringestop. All angle inputs are radians. 

        Parameter
        ---------
        ha : array_like
             The Hour Angle of the source to fringestop too.
        lat : array_like
             The latitude of the observatory.
        dec : array_like
             The declination of the source.
        u : array_like
             The EW separation in wavelengths (increases to the E)
        v : array_like
             The NS separation in wavelengths (increases to the N)
        
        Returns
        -------
        phase : np.ndarray
        The phase required to *correct* the fringeing. Shape is
        given by the broadcast of the arguments together.
        """
        
        uhdotn = np.cos(dec) * np.sin(-ha)
        vhdotn = np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(-ha)
        phase = uhdotn * u + vhdotn * v
        
        return np.exp(2.0J * np.pi * phase)
    

class RFI_Clean(PulsarPipeline):

    always_cut = range(111,138) + range(889,893) + range(856,860) + \
        range(873,877) + range(583,600) + range(552,569) + range(630,645) +\
        range(675,692) + range(753, 768) + range(783, 798) + [267,268,273] + \
        [882, 808, 809, 810, 846, 847, 855, 876, 879, 878, 864, 896, 897, 798, 1017, 978, 977]

    
    def frequency_clean(self, threshold=1e6, broadband_only=True):
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
        if broadband_only==False:
            data = abs(self.data.copy())
            freq = (self.freq)[:, np.newaxis]
            data_freq = data.mean(axis=-1)
            data_freq_norm = data_freq / data_freq.mean() * freq**(4) / freq[-1]**4
            mask = data_freq_norm > threshold

            self.data[np.where(mask)[0], :] = 0.0 

        self.data[self.always_cut, :] = 0.0
        
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
    arr = abs(arr)
    n = radius*2+1
    padded = np.concatenate((arr[:, 1:radius+1][:, ::-1], arr,\
        arr[:, -radius-1:-1][:, ::-1]), axis=1)
    ret = np.cumsum(padded, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    
    return ret[:, n-1:] / n

def find_ongate(arr, ref_prod=[45, 91]):
    """
    For now, take the two 26m autos and find the on-gate after dedispersing and summing
    over whole band.
    
    Parameters
    ==========
    arr: np.array
         Complex array with folded pulsar data. Should be shaped (nfreq, ncorr, ntimes, n_phase_bins)
    ref_prod: list
         List with two 26m autocorrelation product indices

    Return
    ======
    bin_x:
         Array with on-gate for each timestamp, as seen in 26m x-pol auto
    bin_y:
         Array with on-gate for each timestamp, as seen in 26m y-pol auto    
    bin_diff:
         Difference between bin_x and bin_y. Should be zero for most timestamps.
     
    """
    dat_x = abs(arr[:, ref_prod[0]]).mean(axis=0)
    dat_y = abs(arr[:, ref_prod[1]]).mean(axis=0)

    bin_x = np.argmax(dat_x, axis=1)
    bin_y = np.argmax(dat_y, axis=1)

    bin_diff = bin_x - bin_y

    return bin_x, bin_y, bin_diff

def correct_phase_bins(arr, bin_tup):
    """
    Realigns data such that pulse shows up in only one phase bin as a function of time.


    Parameters
    ==========
    arr: np.array
         Complex array with folded pulsar data. Should be shaped (nfreq, ncorr, ntimes, n_phase_bins)
    bin_tup: tuple
         Tuple of length 3 with pulse bin information from ch_pulsar_analysis.find_ongate
    Return
    ======
    arr:
         Pulsar data array realigned such that the pulses are in the central phase bin
    """
    data = arr 
    ntimes = data.shape[2]
    nbins = data.shape[-1]
    bin_x, bin_y, bin_diff = bin_tup
    
    for tt in range(ntimes):
        if bin_diff[tt] == 0:
            on_bin = bin_x[tt]
        elif bin_diff[tt] == 1:
            on_bin = bin_y[tt] #Replace this with something unbiased!
        else:
            on_bin = bin_x[tt-1]
            
        data[:, :, tt] = np.roll(data[:,:,tt], np.int(nbins/2.0) - on_bin, axis=-1)

    return data


def derotate_far(data, RM):
    """
    Undoes Faraday rotation
    """
    freq = np.linspace(800, 400, data.shape[0]) * 1e6
    phase = np.exp(-2j * RM * (3e8 / freq)**2)
    
    if len(data.shape)==3:
        phase = phase[:, np.newaxis]

    return data * phase[:, np.newaxis]

def inj_fake_noise(data, times, cad=5):
    t_ns, p0 = np.linspace(times[0], times[-1], 
              ((times[-1] - times[0]) / cad).astype(int), retstep=True)
    tt = times.repeat(len(t_ns)).reshape(-1, len(t_ns))

    ind_ns = abs(tt - t_ns).argmin(axis=0)
    
    for corr in range(data.shape[1]):
        for k in range(6):
            data[:, corr, ind_ns[:-1]+k] += 0.02 * np.median(data[:, corr], axis=-1)[:, np.newaxis]

    return data, p0

def opt_subtraction(data, phase_axis=-1, time_axis=-2, absval=True):
    """ Performs optimal on/off subtraction on folded data

    Parameters
    ----------
    data : array_like
         Data array of visibilites, with freqs, times, phase
    phase_axis : np.int
         Axis with pulsar phase
    time_axis : np.int
         Time axis
    absval : bool
         Use time avg abs value as weight

    Returns
    -------
    Dynamic spectrum
    """
    if absval:
        data_ = abs(data.copy())
    else:
        data_ = data

    data_sub = data_ - np.mean(data_, axis=phase_axis, keepdims=True)

    data_sub_avg = data_sub.mean(0).mean(0)[np.newaxis, np.newaxis]


    return (data * data_sub_avg).sum(axis=phase_axis)

def common_phasebins(PulsarPipeline, p1, p2, 
                     ngate1, ngate2, on1, on2):

    t1 = PulsarPipeline.phase_bins(p1, 0.0, ngate1)[0]
    t2 = PulsarPipeline.phase_bins(p2, 0.0, ngate2)[0]


    return np.where(t1==on1)[0], np.where(t2==on2)[0]
    
    
