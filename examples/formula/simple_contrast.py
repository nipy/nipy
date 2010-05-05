# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import nipy.testing as niptest
import sympy
from nipy.modalities.fmri import formula, utils, hrf
from nipy.modalities.fmri.fmristat import hrf as delay

c1 = utils.events([3,7,10], f=hrf.glover) # Symbolic function of time
c2 = utils.events([1,3,9], f=hrf.glover) # Symbolic function of time
c3 = utils.events([3,4,6], f=delay.spectral[0])
d = utils.fourier_basis([3,5,7]) # Formula

f = formula.Formula([c1,c2,c3]) + d
contrast = formula.Formula([c1-c2, c1-c3])

t = formula.make_recarray(np.linspace(0,20,50), 't')

X, c = f.design(t, return_float=True, contrasts={'C':contrast})
preC = contrast.design(t, return_float=True)

C = np.dot(np.linalg.pinv(X), preC).T
niptest.assert_almost_equal(C, c['C'])

print C

