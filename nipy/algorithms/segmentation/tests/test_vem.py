#!/usr/bin/env python

import numpy as np

from ..vem import VEM


def test_vem_2d():
    data = np.random.rand(51, 49)
    v = VEM(data, 2)
    v.run()


def test_vem_3d():
    data = np.random.rand(21, 22, 23)
    v = VEM(data, 2)
    v.run()
