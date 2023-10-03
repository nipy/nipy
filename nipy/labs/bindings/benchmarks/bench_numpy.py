# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import time

import numpy as np

from .. import copy_vector


def time_ratio(t0,t1):
    if t1==0:
        return np.inf
    else:
        return t0/t1

def time_copy_vector(x):
    t0 = time.process_time()
    y0 = copy_vector(x, 0)
    dt0 = time.process_time()-t0
    t1 = time.process_time()
    y1 = copy_vector(x, 1)
    dt1 = time.process_time()-t1
    ratio = time_ratio(dt0,dt1)
    print(f'  using fff_array: {dt0:f} sec')
    print(f'  using numpy C API: {dt1:f} sec')
    print(f'  ratio: {ratio:f}')

def bench_copy_vector_contiguous():
    x = (1000*np.random.rand(1_000_000)).astype('int32')
    print('Contiguous buffer copy (int32-->double)')
    time_copy_vector(x)

def bench_copy_vector_strided():
    x0 = (1000*np.random.rand(2_000_000)).astype('int32')
    x = x0[::2]
    print('Non-contiguous buffer copy (int32-->double)')
    time_copy_vector(x)
