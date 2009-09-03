"""
Test the design_matrix utilities.

Note that the tests just looks whether the data produces has correct dimension,
not whether it is exact
"""

import numpy as np
from nipy.neurospin.utils.design_matrix import *

def basic_paradigm():
    conditions = [0,0,0,1,1,1,2,2,2]
    onsets=[30,70,100,10,30,90,30,40,60]
    paradigm = np.vstack(([conditions,onsets])).T
    return paradigm

def block_paradigm():
    conditions = [0,0,0,1,1,1,2,2,2]
    onsets=[30,70,100,10,30,90,30,40,60]
    offsets=[35,75,105,15,35,95,35,45,65]
    paradigm = np.vstack(([conditions,onsets,offsets])).T
    return paradigm

def test_dmtx1():
    """ basic test based on basic_paradigm and caninical hrf
    """
    tr = 1.0
    frametimes = np.linspace(0,127*tr,128)
    paradigm = basic_paradigm()
    hrf_model='Canonical'
    X, names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='Polynomial', order=3)
    print names
    assert(len(names)==7)

def test_dmtx1b():
    """ idem test_dmtx1, but different test
    """
    tr = 1.0
    frametimes = np.linspace(0,127*tr,128)
    paradigm = basic_paradigm()
    hrf_model='Canonical'
    X,names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                        drift_model='Polynomial', order=3)
    print np.shape(X)
    assert(X.shape==(128,7))
    
def test_dmtx2():
    """ idem test_dmtx1 with a different drift term
    """
    tr = 1.0
    frametimes = np.linspace(0,127*tr,128)
    paradigm = basic_paradigm()
    hrf_model='Canonical'
    X,names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                        drift_model='Cosine', hfcut=63)
    print names
    assert(len(names)==8)

def test_dmtx3():
    """ idem test_dmtx1 with a different drift term
    """
    tr = 1.0
    frametimes = np.linspace(0,127*tr,128)
    paradigm = basic_paradigm()
    hrf_model='Canonical'
    X,names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                        drift_model='Blank')
    print names
    assert(len(names)==4)  

def test_dmtx4():
    """ idem test_dmtx1 with a different hrf model
    """
    tr = 1.0
    frametimes = np.linspace(0,127*tr,128)
    paradigm = basic_paradigm()
    hrf_model='Canonical With Derivative'
    X, names= dmtx_light(frametimes, paradigm, hrf_model=hrf_model,
                         drift_model='Polynomial', order=3)
    print names
    assert(len(names)==10)

def test_dmtx5():
    """ idem test_dmtx1 with a block paradigm
    """
    tr = 1.0
    frametimes = np.linspace(0,127*tr,128)
    paradigm = block_paradigm()
    hrf_model='Canonical'
    X, names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='Polynomial', order=3)
    print names
    assert(len(names)==7)

def test_dmtx6():
    """ idem test_dmtx1 with a block paradigm and the hrf derivative
    """
    tr = 1.0
    frametimes = np.linspace(0,127*tr,128)
    paradigm = block_paradigm()
    hrf_model='Canonical With Derivative'
    X, names= dmtx_light(frametimes, paradigm,  hrf_model=hrf_model,
                         drift_model='Polynomial', order=3)
    print names
    assert(len(names)==10)

    
if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

