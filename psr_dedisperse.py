import numpy as np 

nu_min = 0.4 # min frequency in GHz
nu_max = 0.8 
DM = 26.0

N = 505 #Number of pixels per pulse period
n_freq = 1024

A = np.zeros([n_freq, N])
T = 750.0

def freq_pixel(nu):
    return nu_min / n_freq * nu + nu_min

for nu in range(1024):
        t_arrival = 4.15 * DM * (1/(nu_max**2) - 1/freq_pixel(nu)**2) 
        A[nu, mod(np.int(t_arrival * N / T), N)] = 1 # Converts time separation to pixels

A_full = np.hstack((A,A,A,A,A)) 

"""
To de-disperse a pulsar observation the observation length needs to be known as well as the dispersion measure. 
"""
def time_delay(DM, n_freq):
        """ Gets time delay from pulse dispersion. Assumes frequencies are in GHz.
        
        Parameters
        ----------
        DM : float or array of floats
                Pulsar dispersion measure in pc/cm**3
        n_freq :
                Number of delay frequencies 

        Returns
        -------
        delay_nu :
                Difference in arrival time between nu_max and nu (t_nu - t_nu_{max}) 
                in milliseconds
        nu : 
                Frequency vector
        """
        nu = np.linspace(nu_min, nu_max, n_freq) 
        delay_nu = 4.15 * DM * (1/(nu_max**2) - 1/nu**2) 

        return delay_nu, nu

def de_disperse(Data, DM, del_t):
        """ De-disperses pulsar data. First fourier transforms data along time axis then get 
        the fourier frequencies with length "ntimes". The FT is then multiplied by the delay 
        phase factor, correcting for dispersion, and data is transformed back.
        
        Parameters
        ----------
        Data : float or array of floats
                Pulsar data array. Assumes (freq, corr_prod, time)
        DM :
                Pulsar dispersion measure in pc/cm**3 
        del_t: 
                Total observation time in seconds 
        Returns
        -------
        De-dispersed data

        """
        n_times = Data.shape[-1]
        FT_Data = np.fft.fft(np.hanning(n_times)[np.newaxis, :] * Data, axis=-1)
        ft_freq = np.fft.fftfreq(n_times)
        delay_nu = time_delay(DM, Data.shape[0])[0]
        FT_Data = FT_Data * np.exp(2j * np.pi * ft_freq[np.newaxis, :] * delay_nu[:, np.newaxis] * n_times / del_t)

        return np.fft.fft(np.hanning(n_times)[np.newaxis, :] * FT_Data, axis=-1)





