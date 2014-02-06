import numpy as np

class PhaseAnalysis():
    
    def __init__(self, on_gate):
        self.on_gate = on_gate
        self.n_ant = 8
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
      
    def solve_lag(self, lag_pixel):
          
