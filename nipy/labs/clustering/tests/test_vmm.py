"""
Test the Von-Mises-Fisher mixture model

Author : Bertrand Thirion, 2010 
"""

#!/usr/bin/env python

# to run only the simple tests:
# python testClustering.py Test_Clustering

import numpy as np
#from numpy.testing import assert_almost_equal
from ..von_mises_fisher_mixture import (VonMisesMixture,
                                        sphere_density, 
                                        select_vmm, 
                                        select_vmm_cv)


def test_spherical_area():
    """
    test the co_lavbelling functionality
    """
    points, area = sphere_density(100)
    assert (np.abs(area.sum()-4*np.pi)<1.e-2)

def test_von_mises_fisher_density():
    """
    test that a density gets indeed computed on the unit sphere
    for a one-component model (k=1)
    """
    x = np.random.randn(100,3)
    x = (x.T/np.sqrt(np.sum(x**2,1))).T

    for precision in [.1, 1., 10., 100.]:
        k = 1
        vmd = VonMisesMixture(k, precision, null_class=False)
        vmd.estimate(x)
    
        # check that it sums to 1
        s, area = sphere_density(100)
        assert np.abs((vmd.mixture_density(s)*area).sum()-1)<1.e-2

def test_von_mises_fisher_density_plus_null():
    """
    idem test_von_mises_fisher_density, but with a null class
    """
    x = np.random.randn(100,3)
    x = (x.T/np.sqrt(np.sum(x**2,1))).T

    for precision in [.1, 1., 10., 100.]:
        k = 1
        vmd = VonMisesMixture(k, precision, null_class=True)
        vmd.estimate(x)
    
        # check that it sums to 1
        s, area = sphere_density(100)
        assert np.abs((vmd.mixture_density(s)*area).sum()-1)<1.e-2

def test_von_mises_mixture_density():
    """
    test that a density gets indeed computed on the unit sphere
    for a mixture model (k=3)
    """
    x = np.random.randn(100,3)
    x = (x.T/np.sqrt(np.sum(x**2,1))).T

    k = 3
    for precision in [.1, 1., 10., 100.]:
        vmd = VonMisesMixture(k, precision, null_class=False)
        vmd.estimate(x)
    
        # check that it sums to 1
        s, area = sphere_density(100)
        assert np.abs((vmd.mixture_density(s)*area).sum()-1)<1.e-2


def test_von_mises_mixture_density_plus_null():
    """
    idem test_von_mises_mixture_density, but with a null class
    """
    x = np.random.randn(100,3)
    x = (x.T/np.sqrt(np.sum(x**2,1))).T

    k = 3
    for precision in [.1, 1., 10., 100.]:
        vmd = VonMisesMixture(k, precision, null_class=True)
        vmd.estimate(x)
    
        # check that it sums to 1
        s, area = sphere_density(100)
        assert np.abs((vmd.mixture_density(s)*area).sum()-1)<1.e-2

def test_dimension_selection_bic():
    """ Tests whether dimension selection yields correct results
    """
    x1 = [0.6, 0.48, 0.64]
    x2 = [-0.8, 0.48, 0.36]
    x3 = [0.48, 0.64, -0.6]
    x = np.random.randn(200, 3) * .1
    x[:40] += x1
    x[40:150] += x2
    x[150:] += x3
    x = (x.T / np.sqrt(np.sum(x**2, 1))).T

    precision = 100.
    my_vmm = select_vmm(range(1,8), precision, False, x)
    assert my_vmm.k==3

def test_dimension_selection_cv():
    """Tests the dimension selection using cross validation
    """
    x1 = [1, 0, 0]
    x2 = [-1, 0, 0]
    x = np.random.randn(20, 3)*.1
    x[0::2] += x1
    x[1::2] += x2
    x = (x.T / np.sqrt(np.sum(x**2,1))).T

    precision = 50.
    sub = np.repeat(np.arange(10), 2)
    my_vmm = select_vmm_cv(range(1,8), precision, x, cv_index=sub, 
                        null_class=False, ninit=5)
    z = np.argmax(my_vmm.responsibilities(x), 1)
    assert len(np.unique(z))>1
    assert len(np.unique(z))<4

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


