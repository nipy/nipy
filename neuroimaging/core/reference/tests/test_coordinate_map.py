import numpy as np
from neuroimaging.testing import *

import nose.tools

from neuroimaging.core.reference.coordinate_map import CoordinateMap
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.testing import anatfile, funcfile
from neuroimaging.core.api import load_image



def test_identity():
    shape = (30,40,50)
    i = CoordinateMap.identity(['zspace', 'yspace', 'xshape'], shape=shape)
    nose.tools.assert_equals(tuple(i.shape), shape)
    y = i.mapping([3,4,5])
    nose.tools.assert_true(np.allclose(y, np.array([3,4,5])))

def test_identity2():
    shape = (30, 40)
    nose.tools.assert_raises(IndexError, CoordinateMap.identity, ['zspace', 'yspace', 'xspace'], shape)


def test_from_affine():
    a = Affine.identity(2)
    g = CoordinateMap.from_affine('ij', 'xy', a, (20,30))
    nose.tools.assert_equals(g.shape, (20,30))

def test_start_step():
    ''' Test from_start_step '''
    dcs = CoordinateMap.from_start_step('ijk', 'xyz', [5,5,5],[2,2,2], (10,20,30))
    nose.tools.assert_true(np.allclose(dcs.affine, [[2,0,0,5],
                                                    [0,2,0,5],
                                                    [0,0,2,5],
                                                    [0,0,0,1]]))
