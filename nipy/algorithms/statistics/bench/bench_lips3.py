import sys

import numpy as np

from .. import intvol

from ..tests.test_intrinsic_volumes import nonintersecting_boxes, randorth
from numpy.testing import measure


def bench_lip3():
    np.random.seed(20111001)
    phi = intvol.Lips3d
    repeat = 4
    bx_sz = 60
    box1, box2, edge1, edge2 = nonintersecting_boxes((bx_sz,)*3)
    c = np.indices(box1.shape).astype(np.float)
    sys.stdout.flush()
    print 'Box1 %6.2f\n' % measure('phi(c,box1)', repeat),
    print 'Box2 %6.2f\n' % measure('phi(c, box2)', repeat),
    print 'Box1+2 %6.2f\n' % measure('phi(c, box1 + box2)', repeat),
    d = np.random.standard_normal((bx_sz,) * 4)
    U = randorth(p=6)[0:3]
    e = np.dot(U.T, c.reshape((c.shape[0], np.product(c.shape[1:]))))
    e.shape = (e.shape[0],) + c.shape[1:]
    print 'Box1+2 d %6.2f\n' % measure('phi(d, box1 + box2)', repeat),
    print 'Box1+2 e %6.2f\n' % measure('phi(e, box1 + box2)', repeat),
    sys.stdout.flush()
