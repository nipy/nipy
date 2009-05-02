
import csv
from StringIO import StringIO

import scipy.stats
import numpy as np

from nipy.modalities.fmri import formula as F
import nipy.testing as niptest

data = """0.0      1      1      1
    2.0      1      1      2
    1.0      1      1      3
    3.0      1      1      4
    0.0      1      1      5
    2.0      1      1      6
    0.0      1      1      7
    5.0      1      1      8
    6.0      1      1      9
    8.0      1      1     10
    2.0      1      2      1
    4.0      1      2      2
    7.0      1      2      3
   12.0      1      2      4
   15.0      1      2      5
    4.0      1      2      6
    3.0      1      2      7
    1.0      1      2      8
    5.0      1      2      9
   20.0      1      2     10
   15.0      1      3      1
   10.0      1      3      2
    8.0      1      3      3
    5.0      1      3      4
   25.0      1      3      5
   16.0      1      3      6
    7.0      1      3      7
   30.0      1      3      8
    3.0      1      3      9
   27.0      1      3     10
    0.0      2      1      1
    1.0      2      1      2
    1.0      2      1      3
    0.0      2      1      4
    4.0      2      1      5
    2.0      2      1      6
    7.0      2      1      7
    4.0      2      1      8
    0.0      2      1      9
    3.0      2      1     10
    5.0      2      2      1
    3.0      2      2      2
    2.0      2      2      3
    0.0      2      2      4
    1.0      2      2      5
    1.0      2      2      6
    3.0      2      2      7
    6.0      2      2      8
    7.0      2      2      9
    9.0      2      2     10
   10.0      2      3      1
    8.0      2      3      2
   12.0      2      3      3
    3.0      2      3      4
    7.0      2      3      5
   15.0      2      3      6
    4.0      2      3      7
    9.0      2      3      8
    6.0      2      3      9
    1.0      2      3     10"""

# Load some data in -- the above data
# is an example from 
# "Applied Linear Statistical Models" that can be found
# The number of Days in a hospital stay are recorded
# based on the Duration of dialysis treatment
# and the Weight of the patient. These
# two variables are described categorically
# as Duration (1 or 2), Weight (1, 2 or 3)
#
# You can find another copy of the data at
#
# http://www-stat.stanford.edu/~jtaylo/courses/stats191/data/kidney.table

D = []
for row in StringIO(data):
    D.append(map(float, row.split()))
D = F.make_recarray(D, ['Days', 'Duration', 'Weight', 'ID'])

# Create the categorical regressors, known as Factors

f1 = F.Factor('Duration', [1,2])
f2 = F.Factor('Weight', [1,2,3])

twoway = f1 * f2

# The columns of X are 0-1 indicator columns,
# return_float = False yields a recarray 
# with interpretable names

X = twoway.design(D, return_float=False)
niptest.assert_equal(set(X.dtype.names), set(('Duration_1*Weight_1', 'Duration_1*Weight_2', 'Duration_1*Weight_3', 'Duration_2*Weight_1', 'Duration_2*Weight_2', 'Duration_2*Weight_3')))

# If we ask for contrasts, the resulting matrix is
# of dtype np.float

contrasts = {'Duration': f1.main_effect,
             'Weight': f2.main_effect,
             'Interaction': f1.main_effect * f2.main_effect,
             'Duration1': f1.terms[0].formula}
X, cons = twoway.design(D, contrasts=contrasts)

# Fit the model

beta = np.dot(np.linalg.pinv(X), D['Days'])
XTXinv = np.linalg.pinv(np.dot(X.T, X))
resid = D['Days'] - np.dot(X, beta)
df_resid = (X.shape[0] - X.shape[1]) # residual degrees of freedom
sigmasq = (resid**2).sum() / df_resid

SS = {}
MS = {}
F = {}
df = {}
p = {}
for n, c in cons.items():
    cbeta = np.dot(c, beta)
    cov_cbeta = np.dot(c, np.dot(XTXinv, c.T))
    if c.ndim > 1:
        df[n] = c.shape[0]
        SS[n] = np.dot(cbeta, np.dot(np.linalg.pinv(cov_cbeta), cbeta))
        MS[n] = SS[n] / df[n]
        F[n] = MS[n] / sigmasq
    else:
        df[n] = 1
        SS[n] = (cbeta**2).sum() / cov_cbeta
        MS[n] = SS[n] / df[n]
        F[n] = MS[n] / sigmasq
    p[n] = scipy.stats.f.sf(F[n], df[n], df_resid)

"""
Output of R:
-----------


> anova(lm(Days~Duration*Weight, X))
Analysis of Variance Table

Response: Days
                Df  Sum Sq Mean Sq F value    Pr(>F)    
Duration         1  209.07  209.07  7.2147  0.009587 ** 
Weight           2  760.43  380.22 13.1210 2.269e-05 ***
Duration:Weight  2  109.03   54.52  1.8813  0.162240    
Residuals       54 1564.80   28.98                      
---
"""

try:
    import rpy
    rpy.r("""
X = read.table('http://www-stat.stanford.edu/~jtaylo/courses/stats191/data/kidney.table', header=T)
names(X)
X$Duration = factor(X$Duration)
X$Weight = factor(X$Weight)
lm(Days~Duration*Weight, X)
A = anova(lm(Days~Duration*Weight, X))
""")
    r = rpy.r('A')
    n = rpy.r('rownames(A)')
    pairs = [(n.index('Duration'), 'Duration'),
             (n.index('Weight'), 'Weight'),
             (n.index('Duration:Weight'), 'Interaction')]

    for i, j in pairs:
        niptest.assert_almost_equal(F[j], r['F value'][i])
        niptest.assert_almost_equal(p[j], r['Pr(>F)'][i])
        niptest.assert_almost_equal(MS[j], r['Mean Sq'][i])
        niptest.assert_almost_equal(df[j], r['Df'][i])
        niptest.assert_almost_equal(SS[j], r['Sum Sq'][i])
except ImportError:
    pass


