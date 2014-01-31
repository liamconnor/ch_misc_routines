import numpy as np
import ch_util.andata
import ch_util.ephemeris as eph
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

    return data, Vis, ctime, RA

def autocorr_ind(nfeeds):
    feeds = np.arange(0, nfeeds)
    return nfeeds * feeds - 0.5 * feeds * (feeds - 1)

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
