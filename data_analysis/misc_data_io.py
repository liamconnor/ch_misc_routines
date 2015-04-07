import numpy as np
import ch_util.andata
import ch_util.ephemeris as eph
import h5py
#import matplotlib.pyplot as plt

chand = ch_util.andata.AnData()

freq_lower = 400.0
freq_upper = 800.0 
n_freq = 1024.0

def get_data(file, start=False, stop=False):
    if not start and not stop:
        data = chand.from_acq_h5(file)
    else:
        data = chand.from_acq_h5(file, start=start, stop=stop)

    Vis = data.vis
    print "vis shape:", Vis.shape

    ctime = data.timestamp
    
    if np.isnan(ctime).sum() > 0:
        print "Removing NaNs"
        Vis[:, :, ctime != ctime] = 0.0
        ctime[ctime != ctime] = 0.0

    RA = eph.transit_RA(ctime)
    fpga_count = data.datasets['timestamp_fpga_count'][:]

    return data, Vis, ctime, RA, fpga_count

def feed_map(feed_i, feed_j, n_feed):
    if feed_i > feed_j:
        return ((2*n_feed*feed_j - feed_j**2 + feed_j)/2) + (feed_i - feed_j)
    else:
        return ((2*n_feed*feed_i - feed_i**2 + feed_i)/2) + (feed_j - feed_i)

def gen_corr_matrix(data, n_feed):
    corr_mat = np.zeros((n_feed, n_feed), np.complex128)

    for feed_i in range(n_feed):
        for feed_j in range(feed_i, n_feed):
                corr_mat[feed_i, feed_j] = data[feed_map(feed_i, feed_j, n_feed)]
                corr_mat[feed_j, feed_i] = np.conj(data[feed_map(feed_i, feed_j, n_feed)])
    
    return corr_mat

def corr2baseline(nfeeds, spacing):
    """ Assumes feed layout on cassettes is the same as correlation ordering,
    i.e. corr[1] is outermost feed correlated with adjacent

    Parameters
    ----------
    nfeeds: 
           Number of single-pol feeds
    spacing:
           Baseline spacing in units of min spacing
    """
    feeds = np.arange(0, nfeeds)
    corr_ind = nfeeds * feeds - 0.5 * feeds * (feeds - 1) + spacing
    return corr_ind[:nfeeds - spacing]

def ind2freq(freq_ind):
    if freq_ind > n_freq - 1 or freq_ind < 0:
        print "You're out of our band!"
    else:
        return freq_upper - freq_ind/1024. * freq_lower

def freq2ind(freq):
    if freq > freq_upper or freq < freq_lower:
        print "You're out of our band!"
    else:
        return np.int(np.round(n_freq * (freq_upper - freq) / freq_lower))

def baselines(pos_arr, n):
    """
    Takes n feed positions and get ew and ns baselines
    """
    pos_ew = pos_arr[:, 0]
    pos_ns = pos_arr[:, 1]

    lew = ew[np.newaxis] * np.ones([n,n]) - ew[:, np.newaxis] * np.ones([n,n])
    lns = ns[np.newaxis] * np.ones([n,n]) - ns[:, np.newaxis] * np.ones([n,n])

    d_ew = lew[np.triu_indices(n)]
    d_ns = lns[np.triu_indices(n)]

    return np.vstack((d_ew, d_ns))

def fft_data(arr, ft='lag'):
    """
    Returns the windowed fft of a time/freq array along 
    the time axis, freq axis, or both.
    
    Parameters
    ==========
    arr: complex array
         (nfreq x ntimes) array of complex visibilities
    ft: str
         transform to return, either lag, m, or mlag

    Returns
    =======
    Returns the fft array that was asked for in ft argument.
    """
    freq_window = np.hanning(arr.shape[0])[:, np.newaxis]  * 0.0 + 1.0
    time_window = np.hanning(arr.shape[-1])[np.newaxis, :] * 0.0 + 1.0

    if ft=='lag':
        return np.fft.fftshift(np.fft.fft(freq_window * arr, axis=0), axes=0)
    elif ft=='m':
        return np.fft.fftshift(np.fft.fft(time_window * arr, axis=-1), axes=-1)
    elif ft=='mlag':
        return np.fft.fftshift(np.fft.fft2(freq_window * time_window * arr))
    else:
        raise Exception('only lag, m, or mlag allowed')

def gain_time(data, n, save_as=False):
    data_mean = data.mean(axis=0)
    gd = np.zeros((n, data_mean.shape[-1]), np.complex128)

    for i in range(data_mean.shape[-1]):
        A = gen_corr_matrix(data_mean[:, i], n)
        eval, evec = np.linalg.eigh(A)
        gd[:, i] = eval[-1]**0.5 * evec[:, -1] 
        
    if save_as:
        f = h5py.File(save_as,'w')
        f.create_dataset('gain_time', data=gd)
        f.close()
        
    return gd
    
def iterate_sol(data, nfeed):
    auto_ind = []
    [auto_ind.append(feed_map(i,i,16)) for i in range(16)]
    print auto_ind

    for ii in range(2):
        print "Iteration %d" % ii
        gain_arr, eval_arr = solve_gain(data, nfeed)
        data[:, auto_ind] = abs(gain_arr)**2
        
    return gain_arr, eval_arr

def solve_gain(data, nfeed, save_as=False):
    gain_arr = np.zeros([data.shape[0], nfeed, 2, data.shape[-1]], np.complex128)
    eval_arr = np.zeros([data.shape[0], nfeed, data.shape[-1]], np.float64)

    for nu in range(data.shape[0]):
        if (nu%64)==0:
            print "Freq %d" % nu
        for tt in range(data.shape[-1]):
            corr_arr = gen_corr_matrix(data[nu, :, tt], nfeed)
            corr_arr[np.diag_indices(nfeed)] = 0.0
            corr_arr[:,9] = 0.0
            corr_arr[9] = 0.0 

            evl, evec = np.linalg.eigh(corr_arr)

#            if tt==data.shape[-1]/2:
#                print evl[-1] / evl[-2]
            
            #eval_arr[nu, tt] = evl[-1] / evl[-2]
            eval_arr[nu, :, tt] = evl
            gain_arr[nu, :, 0, tt] = evl[-1]**0.5 * evec[:, -1]
            gain_arr[nu, :, 1, tt] = evl[-2]**0.5 * evec[:, -2]
            gain_arr *= np.sign(gain_arr[:,0,np.newaxis]) # Demand that the first antenna is positive

    return gain_arr, eval_arr        
            

def imli(arr, vmax=None, vmin=None):
    if not vmax:
        vmax=arr.max()
        vmin=arr.min()
    plt.imshow(arr, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

def calc_baseline(feed_loc):
    """
    Get baselines given an array of feed positions.
    
    Parameters
    ==========
    feed_loc: np.array
            Should be (n,2) arr with x_i, y_i 
            
    Return
    ======
    del_x:
         east-west baselines
    del_y:
         north-south baselines
    d:
         baseline length
    """

    x_pos = feed_loc[:, 0]
    y_pos = feed_loc[:, 1]

    xmat = (x_pos * np.ones([len(x_pos), len(x_pos)]))
    ymat = (y_pos * np.ones([len(y_pos), len(y_pos)]))

    del_x, del_y = (xmat - xmat.transpose())[np.triu_indices(len(x_pos))], (ymat - ymat.transpose())[np.triu_indices(len(y_pos))]
    
    return del_x, del_y, np.sqrt(del_x**2 + del_y**2)
    
def svd_model(arr, phase_only=True):
    """
    Take time/freq visibilities SVD, zero out all but the largest mode, multiply original data by complex conjugate
    
    Parameters
    ----------
    arr : array_like
       Time/freq visiblity matrix 

    Returns
    -------
    Original data array multiplied by the largest SVD mode conjugate
    """

    u,s,w = np.linalg.svd(arr)
    s[1:] = 0.0
    S = np.zeros([len(u), len(w)], np.complex128)
    S[:len(s), :len(s)] = np.diag(s)
    
    print u.shape, S.shape, w.shape

    model = np.dot(np.dot(u, S), w)

    if phase_only==True:
        return arr * np.exp(-1j * np.angle(model))
    else:
        return arr / (model)        


def correct_delay(data, nfreq=1024):
    lag_max = np.argmax(abs(np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)).mean(-1))
    os_pix = lag_max - data.shape[0]/2 
    phase = np.exp(-2j*np.pi * os_pix * np.arange(nfreq)/nfreq)
    
    return data * phase[:, np.newaxis]

def pulse_weight(data, weights):
    """ Multiply each pulse by its flux, normalize by sum
    
    Parameters
    ----------
    data : array_like
         (nfreq x ntimes) data to be weighted
    weights : array_like
         ntimes-length vector with pulse weights

    Returns
    -------
    Weighted data
    """
    return data * weights[np.newaxis] / weights.sum()

    
def reshape_zeros(arr, frb=4, trb=2):
    """ Bin down in time and frequency, accounting for the 
    zerod channels 

    Parameters
    ----------
    arr : array_like
        (nfreq, -1, ntimes) array
    frb : np.int
        frequency rebin
    trb : np.int
        time rebin

    Return 
    ------
    Rebinned array
    """

    nfreq = arr.shape[0] / frb
    ntimes = arr.shape[-1] / trb

    Arr = arr.reshape(nfreq, frb, -1, ntimes, trb).sum(1).sum(-1)

    freq_w = (arr.mean(-1) != 0.0).reshape(nfreq, frb, -1).sum(1)
    time_w = (arr.mean(0) != 0.0).reshape(-1, ntimes, trb).sum(-1)

    return Arr / freq_w[:, :, np.newaxis] / time_w[np.newaxis]


def gen_delay_tab(delays):
    """ Takes delays for n-feeds and returns relative delays 
    for n*(n+1)/2 correlations
    """

    n = len(delays)
    del_arr = delays.repeat(n).reshape(-1, n)

    return (del_arr - del_arr.T)[np.triu_indices(n)]

def running_mean_rfi(data, n=10, beam_ra=None):
    """ One dimensional rfi search
    """

    m = len(data)
    data = data[:(m//n) * n]

    data_rm = data.reshape(-1, n).mean(-1).repeat(n)
    r = (data / data_rm)

    if beam_ra != None:
        r[beam_ra] = 1.0
        
    mask = np.where(r > 3.0)[0]

    return mask
