import numpy as np
import misc_data_io as misc

class PhaseAnalysis():
    
    def __init__(self, on_gate):
        self.on_gate = on_gate
        self.n_feed = 8
        self.ncorr = self.n_feed * (self.n_feed + 1) / 2

    def get_lag_pixel(self, data):
        """
        Takes an fft along freq axis after subtracting adjacent off-gates from ongate.
        Method then finds the maximum lag pixel and saves down each correlation's lag.
        
        Parameters
        ----------
        data: complex arr
              Array with gated pulsar data whose shape is either (nfreq,ncorr,ntime,nbin)
              or (nfreq, ntime, nbin)
        Returns
        -------
        lag_pixel: vector
              vector that stores lag pixel for each correlation product in data
        """

        if len(data.shape)==3:
            data = data[:, np.newaxis, :]
        elif len(data.shape)!=4:
            raise Exception("data has wrong shape")
        
        nfreq = data.shape[0]
        data_pulse = data[:, :, :, self.on_gate] - 0.5 * (data[:, :, :, self.on_gate-2] + data[:, :, :, self.on_gate+2])
        data_lag = abs(np.fft.fftshift(np.fft.fft(np.hanning(nfreq)[:, np.newaxis, np.newaxis]\
                                                      * data_pulse, axis=0), axes=0)).mean(axis=-1)
        lag_pixel = np.zeros([data.shape[1]])
        for corr in range(data.shape[1]):
            max_pixel = np.where(data_lag[:, corr] == data_lag[:, corr].max())[0][0] - nfreq/2.
            lag_pixel[corr] = max_pixel

        return lag_pixel


    def solve_lag(self, lag_pixel):
        """
        Exponentiates lags and puts them into an n_feedxn_feed matrix. 
        Eigendecomposes to solve for exponential lag vectors

        Parameters
        ----------
        lag_pixel: vector 
                stores each correlation's lag pixel

        Returns 
        -------
        lag_sol: vector
                solution for lag in pixels for all n_feed antennas
        """
        lag_exp = np.exp(1j * lag_pixel / 100.0) 
        # divide by 100. so exponentials don't tempt numerical precision
        phi_arr = misc.gen_corr_matrix(lag_exp, self.n_feed)
        eval, evec = np.linalg.eigh(phi_arr)

        phi = eval[-1]**0.5 * evec[:, -1]
        lag_sol = np.angle(phi) * 100.0

        return lag_sol
