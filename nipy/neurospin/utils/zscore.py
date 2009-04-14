import numpy as np
import scipy.stats

TINY = 1e-15

def zscore(pvalue): 
    """ Return the corresponding Z score for a given pvalue.
    """
    pvalue = np.minimum(np.maximum(pvalue, TINY), 1.-TINY)
    z = scipy.stats.norm.isf(pvalue)
    return z
