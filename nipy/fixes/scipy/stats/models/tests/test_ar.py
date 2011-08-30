
from numpy.testing import assert_array_equal

from .exampledata import x, y
from .. import regression

# FIXME: This test does not test any values
# TODO: spend an hour or so to create a test like test_ols.py
# with R's output, the script and the data used for the script
#
# Although, it should be said that this, in R
# x = as.matrix(read_table('x.csv'))
# y = as.matrix(read_table('y.csv'))
# res = arima(y, xreg=x, order=c(2,0,0))
#
# gives an error ``system is computationally singular``

def test_armodel():
    for i in range(1,4):
        model = regression.ARModel(x, i)
        for i in range(20):
            results = model.fit(y)
            rho, sigma = regression.yule_walker(y - results.predicted)
            model = regression.ARModel(model.design, rho)
        print "AR coefficients:", model.rho

