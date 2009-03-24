import numpy as np
import sympy
from neuroimaging.modalities.fmri import formula, utils, hrf

c1 = utils.events([3,7,10], f=hrf.glover_sympy) # Symbolic function of time
c2 = utils.events([1,3,9], f=hrf.glover_sympy) # Symbolic function of time
d = utils.fourier_basis([3,5,7]) # Formula

f = formula.Formula([c1,c2]) + d

# cleaner way

h = sympy.Function('hrf')
h2 =sympy.Function('hrf2')
h3 =sympy.Function('hrf3')
c1 = utils.events([3,7,10], f=h)
e1 = utils.events([3,7,10], f=h2)
c2 = utils.events([1,3,9], f=h)
e2 = utils.events([3,7,10], f=h3)
c3 = utils.events([2,4,8], f=h)
d = utils.fourier_basis([3,5,7]) # Formula

f = formula.Formula([c1,e1,c2,e2,c3]) + d
f.aliases['hrf'] = hrf.glover

contrast = formula.Formula([c1-c2, c1-c3])
contrast.aliases['hrf'] = hrf.glover

tval = np.linspace(0,20,50).view(np.dtype([('t', np.float)]))

d = formula.Design(f, return_float=True)
X = d(tval)

d2 = formula.Design(contrast, return_float=True)
preC = d2(tval)

C = np.dot(np.linalg.pinv(X), preC)
print C
