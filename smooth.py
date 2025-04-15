import numpy as np

def f_smooth(TP, span):
    # TP: year, lat, lon
    a, b, c = TP.shape
    TPS = np.zeros((a-span+1, b, c))
    for jj in range(span):
        TPS += TP[jj:a-span+jj+1, :, :]
    TPS /= span
    return TPS