import numpy as np

def sig_cut(Data):
    threshold = 0.5

    pol_ratio = Data[:, 1, :].sum(axis=-1) / np.sqrt( Data[:, 0, :].sum(axis=-1) * Data[:, 8, :].sum(axis=-1))

    Data[pol_ratio > 0.5, :, :] = 0.0

    return Data
