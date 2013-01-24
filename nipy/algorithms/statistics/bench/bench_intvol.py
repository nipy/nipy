import sys

import numpy as np

from .. import intvol

from ..tests.test_intrinsic_volumes import nonintersecting_boxes, randorth

import numpy.testing as npt

def bench_lips3d():
    np.random.seed(20111001)
    phi = intvol.Lips3d
    EC3d = intvol.EC3d
    repeat = 4
    bx_sz = 60
    box1, box2, edge1, edge2 = nonintersecting_boxes((bx_sz,)*3)
    c = np.indices(box1.shape).astype(np.float)
    sys.stdout.flush()
    print "\nIntrinsic volumes 3D"
    print "--------------------"
    print 'Box1 %6.2f\n' % npt.measure('phi(c,box1)', repeat),
    print 'Box2 %6.2f\n' % npt.measure('phi(c, box2)', repeat),
    print 'Box1+2 %6.2f\n' % npt.measure('phi(c, box1 + box2)', repeat),
    d = np.random.standard_normal((10,) + (bx_sz,) * 3)
    print 'Box1+2 d %6.2f\n' % npt.measure('phi(d, box1 + box2)', repeat),
    U = randorth(p=6)[0:3]
    e = np.dot(U.T, c.reshape((c.shape[0], -1)))
    e.shape = (e.shape[0],) + c.shape[1:]
    print 'Box1+2 e %6.2f\n' % npt.measure('phi(e, box1 + box2)', repeat),
    print 'Box1+2 EC %6.2f\n' % npt.measure('EC3d(box1 + box2)', repeat),
    sys.stdout.flush()


def bench_lips2d():
    np.random.seed(20111001)
    phi = intvol.Lips2d
    EC2d = intvol.EC2d
    repeat = 4
    bx_sz = 500
    box1, box2, edge1, edge2 = nonintersecting_boxes((bx_sz,)*2)
    c = np.indices(box1.shape).astype(np.float)
    sys.stdout.flush()
    print "\nIntrinsic volumes 2D"
    print "--------------------"
    print 'Box1 %6.2f\n' % npt.measure('phi(c,box1)', repeat),
    print 'Box2 %6.2f\n' % npt.measure('phi(c, box2)', repeat),
    print 'Box1+2 %6.2f\n' % npt.measure('phi(c, box1 + box2)', repeat),
    d = np.random.standard_normal((10,) + (bx_sz,) * 2)
    print 'Box1+2 d %6.2f\n' % npt.measure('phi(d, box1 + box2)', repeat),
    U = randorth(p=6)[0:2]
    e = np.dot(U.T, c.reshape((c.shape[0], -1)))
    e.shape = (e.shape[0],) + c.shape[1:]
    print 'Box1+2 e %6.2f\n' % npt.measure('phi(e, box1 + box2)', repeat),
    print 'Box1+2 EC %6.2f\n' % npt.measure('EC2d(box1 + box2)', repeat),
    sys.stdout.flush()


def bench_lips1d():
    np.random.seed(20111001)
    phi = intvol.Lips1d
    EC1d = intvol.EC1d
    repeat = 4
    bx_sz = 100000
    box1, box2, edge1, edge2 = nonintersecting_boxes((bx_sz,))
    c = np.indices(box1.shape).astype(np.float)
    sys.stdout.flush()
    print "\nIntrinsic volumes 1D"
    print "--------------------"
    print 'Box1 %6.2f\n' % npt.measure('phi(c,box1)', repeat),
    print 'Box2 %6.2f\n' % npt.measure('phi(c, box2)', repeat),
    print 'Box1+2 %6.2f\n' % npt.measure('phi(c, box1 + box2)', repeat),
    d = np.random.standard_normal((10, bx_sz))
    print 'Box1+2 d %6.2f\n' % npt.measure('phi(d, box1 + box2)', repeat),
    U = randorth(p=6)[0:1]
    e = np.dot(U.T, c.reshape((c.shape[0], -1)))
    e.shape = (e.shape[0],) + c.shape[1:]
    print 'Box1+2 e %6.2f\n' % npt.measure('phi(e, box1 + box2)', repeat),
    print 'Box1+2 EC %6.2f\n' % npt.measure('EC1d(box1 + box2)', repeat),
    sys.stdout.flush()
