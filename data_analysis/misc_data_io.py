import numpy as np
import ch_util.andata
import ch_util.ephemeris as eph
import h5py
import matplotlib.pyplot as plt

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
    Returns the windowed fft of a time/freq array along the time axis, freq axis, or both.
    
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
    freq_window = np.hanning(arr.shape[0])[:, np.newaxis] 
    time_window = np.hanning(arr.shape[-1])[np.newaxis, :]

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
    
