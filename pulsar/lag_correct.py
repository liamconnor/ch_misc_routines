import numpy as np
import misc_data_io as misc

class PhaseAnalysis():
    
    def __init__(self, on_gate):
        self.on_gate = on_gate
        self.n_feed = 8
        self.ncorr = 0.5 * n_ant * (n_ant + 1)

    def get_lag_pixel(self, data):

        if len(data.shape)==3:
            data = data[:, np.newaxis, :]
        elif len(data.shape)!=4:
            raise Exception("data has wrong shape")
        
        nfreq = data.shape[0]
        data_pulse = data[:, :, :, ongate] - 0.5 * (data[:, :, :, self.on_gate-2] + data[:, :, :, self.ongate+2])
        data_lag = abs(np.fft.fftshift(np.fft.fft(np.hanning(nfreq)[:, np.newaxis, np.newaxis]\
                                                      * data_pulse, axis=0), axes=0)).mean(axis=-1)
        lag_pixel = np.zeros([data.shape[0])
        for corr in range(self.ncorr):
            max_pixel = np.where(data_lag[:, corr] == data_lag[:, corr].max())[0][0] - nfreq/2.
            lag_pixel[corr] = max_pixel

        return lag_pixel


    def solve_lag(self, lag_pixel):
        lag_exp = np.exp(2 * np.pi * 1j * lag_pixel / 100.0) 
        # divide by 100. so exponentials don't tempt numerical precision
        phi_arr = misc.gen_corr_matrix(lag_exp, self.n_feed)
        eval, evec = np.linalg.eigh(phi_arr)

        phi = eval[-1]**0.5 * evec[:, -1]
        lag_sol = np.angle(phi) * 100.0

        return lag_sol
