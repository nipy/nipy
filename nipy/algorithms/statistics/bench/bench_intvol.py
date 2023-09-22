# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import sys

import numpy as np
import numpy.testing as npt

from .. import intvol
from ..tests.test_intrinsic_volumes import nonintersecting_boxes, randorth


def bench_lips3d():
    np.random.seed(20111001)
    phi = intvol.Lips3d
    EC3d = intvol.EC3d
    repeat = 4
    bx_sz = 60
    box1, box2, edge1, edge2 = nonintersecting_boxes((bx_sz,)*3)
    c = np.indices(box1.shape).astype(np.float64)
    sys.stdout.flush()
    print("\nIntrinsic volumes 3D")
    print("--------------------")
    print(f"Box1 {npt.measure('phi(c,box1)', repeat):6.2f}")
    print(f"Box2 {npt.measure('phi(c, box2)', repeat):6.2f}")
    print(f"Box1+2 {npt.measure('phi(c, box1 + box2)', repeat):6.2f}")
    d = np.random.standard_normal((10,) + (bx_sz,) * 3)
    print(f"Box1+2 d {npt.measure('phi(d, box1 + box2)', repeat):6.2f}")
    U = randorth(p=6)[0:3]
    e = np.dot(U.T, c.reshape((c.shape[0], -1)))
    e.shape = (e.shape[0],) + c.shape[1:]
    print(f"Box1+2 e {npt.measure('phi(e, box1 + box2)', repeat):6.2f}")
    print(f"Box1+2 EC {npt.measure('EC3d(box1 + box2)', repeat):6.2f}")
    sys.stdout.flush()


def bench_lips2d():
    np.random.seed(20111001)
    phi = intvol.Lips2d
    EC2d = intvol.EC2d
    repeat = 4
    bx_sz = 500
    box1, box2, edge1, edge2 = nonintersecting_boxes((bx_sz,)*2)
    c = np.indices(box1.shape).astype(np.float64)
    sys.stdout.flush()
    print("\nIntrinsic volumes 2D")
    print("--------------------")
    print(f"Box1 {npt.measure('phi(c,box1)', repeat):6.2f}")
    print(f"Box2 {npt.measure('phi(c, box2)', repeat):6.2f}")
    print(f"Box1+2 {npt.measure('phi(c, box1 + box2)', repeat):6.2f}")
    d = np.random.standard_normal((10,) + (bx_sz,) * 2)
    print(f"Box1+2 d {npt.measure('phi(d, box1 + box2)', repeat):6.2f}")
    U = randorth(p=6)[0:2]
    e = np.dot(U.T, c.reshape((c.shape[0], -1)))
    e.shape = (e.shape[0],) + c.shape[1:]
    print(f"Box1+2 e {npt.measure('phi(e, box1 + box2)', repeat):6.2f}")
    print(f"Box1+2 EC {npt.measure('EC2d(box1 + box2)', repeat):6.2f}")
    sys.stdout.flush()


def bench_lips1d():
    np.random.seed(20111001)
    phi = intvol.Lips1d
    EC1d = intvol.EC1d
    repeat = 4
    bx_sz = 100000
    box1, box2, edge1, edge2 = nonintersecting_boxes((bx_sz,))
    c = np.indices(box1.shape).astype(np.float64)
    sys.stdout.flush()
    print("\nIntrinsic volumes 1D")
    print("--------------------")
    print(f"Box1 {npt.measure('phi(c,box1)', repeat):6.2f}")
    print(f"Box2 {npt.measure('phi(c, box2)', repeat):6.2f}")
    print(f"Box1+2 {npt.measure('phi(c, box1 + box2)', repeat):6.2f}")
    d = np.random.standard_normal((10, bx_sz))
    print(f"Box1+2 d {npt.measure('phi(d, box1 + box2)', repeat):6.2f}")
    U = randorth(p=6)[0:1]
    e = np.dot(U.T, c.reshape((c.shape[0], -1)))
    e.shape = (e.shape[0],) + c.shape[1:]
    print(f"Box1+2 e {npt.measure('phi(e, box1 + box2)', repeat):6.2f}")
    print(f"Box1+2 EC {npt.measure('EC1d(box1 + box2)', repeat):6.2f}")
    sys.stdout.flush()
