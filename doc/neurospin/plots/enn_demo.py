import numpy as np
x = np.c_[np.random.normal(size=1e4), 
          np.random.normal(scale=4, size=1e4)]

from neuroimaging.neurospin.utils.emp_null import ENN
enn = ENN(x)
enn.threshold(verbose=True)

