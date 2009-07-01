
from numpy.testing import *

from exampledata import x, y
import nipy.fixes.scipy.stats.models as SSM

# FIXME: This test does not test any values
# TODO: spend an hour or so to create a test like test_ols.py
# with R's output, the script and the data used for the script

def test_armodel():
    for i in range(1,4):
        model = SSM.regression.ARModel(x, i)
        for i in range(20):
            results = model.fit(y)
            rho, sigma = SSM.regression.yule_walker(y - results.predicted)
            model = SSM.regression.ARModel(model.design, rho)
        print "AR coefficients:", model.rho

