import numpy as N

from scipy.stats import median

def MAD(a, c=0.6745):
    a = N.asarray(a, N.float64)
    d = median(a, axis=0)
    d.shape = (1,) + a.shape[1:]
    return median(N.fabs(a - d) / c)

