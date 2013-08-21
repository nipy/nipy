# EMAcs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from ...api import write_data, slice_generator
from .. import generators as gen

from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_almost_equal, assert_array_equal

shape = (10,20,30)
DATA = np.zeros(shape)
DATA2 = np.ones(shape)
shape = (3,5,4)
DATA3 = np.zeros(shape)


def test_read_slices():
    for _, d in slice_generator(DATA):
        assert_equal(d.shape, (20, 30))
    for _, d in slice_generator(DATA, axis=1):
        assert_equal(d.shape, (10, 30))
    for _, d in slice_generator(DATA, axis=2):
        assert_equal(d.shape, (10, 20))


def test_write_slices():
    tmp = np.zeros(DATA.shape)
    write_data(tmp, slice_generator(DATA))
    assert_almost_equal(tmp, np.asarray(DATA))
    tmp = np.zeros(DATA.shape)
    write_data(tmp, slice_generator(DATA, axis=1))
    assert_almost_equal(tmp, np.asarray(DATA))
    tmp = np.zeros(DATA.shape)
    write_data(tmp, slice_generator(DATA, axis=2))
    assert_almost_equal(tmp, np.asarray(DATA))


def test_multi_slice():
    for _, d in slice_generator(DATA, axis=[0, 1]):
        assert_equal(d.shape, (30,))
    for _, d in slice_generator(DATA, axis=[2, 1]):
        assert_equal(d.shape, (10,))


def test_multi_slice_write():
    a = np.zeros(DATA.shape)
    write_data(a, slice_generator(DATA, axis=[0, 1]))
    assert_almost_equal(a, np.asarray(DATA))


def test_parcel():
    parcelmap = np.zeros(DATA3.shape)
    parcelmap[0,0,0] = 1
    parcelmap[1,1,1] = 1
    parcelmap[2,2,2] = 1
    parcelmap[1,2,1] = 2
    parcelmap[2,3,2] = 2
    parcelmap[0,1,0] = 2
    parcelseq = (0, 1, 2, 3)
    expected = [np.product(DATA3.shape) - 6, 3, 3, 0]
    iterator = gen.data_generator(DATA3,
                                  gen.parcels(parcelmap, labels=parcelseq))
    for i, pair in enumerate(iterator):
        s, d = pair
        assert_equal((expected[i],), d.shape)
    iterator = gen.data_generator(DATA3, gen.parcels(parcelmap))
    for i, pair in enumerate(iterator):
        s, d = pair
        assert_equal((expected[i],), d.shape)


def test_parcel_exclude():
    # Test excluding from parcels
    data = np.arange(5)
    ps = gen.parcels(data, (1, 3))
    assert_array_equal(next(ps), [False, True, False, False, False])
    assert_array_equal(next(ps), [False, False, False, True, False])
    assert_raises(StopIteration, next, ps)
    ps = gen.parcels(data, (1, 3), exclude=(1,))
    assert_array_equal(next(ps), [False, False, False, True, False])
    assert_raises(StopIteration, next, ps)
    ps = gen.parcels(data, (1, 3), exclude=(3,))
    assert_array_equal(next(ps), [False, True, False, False, False])
    assert_raises(StopIteration, next, ps)
    ps = gen.parcels(data, (1, 3), exclude=(3, 1))
    assert_raises(StopIteration, next, ps)
    # Test that two element exclude works
    ps = gen.parcels(data, (1, 3, 4), exclude=(1, 4))
    assert_array_equal(next(ps), [False, False, False, True, False])
    assert_raises(StopIteration, next, ps)
    # Also as np.array
    ps = gen.parcels(data, (1, 3, 4), exclude=np.array((1, 4)))
    assert_array_equal(next(ps), [False, False, False, True, False])
    assert_raises(StopIteration, next, ps)
    # Test that parcels continue to be returned in sorted order
    rng = np.random.RandomState(42)
    data = rng.normal(size=(10,))
    uni = np.sort(np.unique(data)) # Should already be sorted in fact
    values = [np.mean(data[p]) # should be scalar anyway
              for p in gen.parcels(data, exclude=uni[0:2])]
    assert_array_equal(values, uni[2:])


def test_parcel_write():
    parcelmap = np.zeros(DATA3.shape)
    parcelmap[0,0,0] = 1
    parcelmap[1,1,1] = 1
    parcelmap[2,2,2] = 1
    parcelmap[1,2,1] = 2
    parcelmap[2,3,2] = 2
    parcelmap[0,1,0] = 2
    parcelseq = (0, 1, 2, 3)
    expected = [np.product(DATA3.shape) - 6, 3, 3, 0]
    iterator = gen.parcels(parcelmap, labels=parcelseq)
    for i, s in enumerate(iterator):
        value = np.arange(expected[i])
        DATA3[s] = value
    iterator = gen.parcels(parcelmap, labels=parcelseq)
    for i, pair in enumerate(gen.data_generator(DATA3, iterator)):
        s, d = pair
        assert_equal((expected[i],), d.shape)
        assert_array_equal(d, np.arange(expected[i]))
    iterator = gen.parcels(parcelmap)
    for i, s in enumerate(iterator):
        value = np.arange(expected[i])
        DATA3[s] = value
    iterator = gen.parcels(parcelmap)
    for i, pair in enumerate(gen.data_generator(DATA3, iterator)):
        s, d = pair
        assert_equal((expected[i],), d.shape)
        assert_array_equal(d, np.arange(expected[i]))


def test_parcel_copy():
    parcelmap = np.zeros(DATA3.shape)
    parcelmap[0,0,0] = 1
    parcelmap[1,1,1] = 1
    parcelmap[2,2,2] = 1
    parcelmap[1,2,1] = 2
    parcelmap[2,3,2] = 2
    parcelmap[0,1,0] = 2
    parcelseq = (0, 1, 2, 3)
    expected = [np.product(DATA3.shape) - 6, 3, 3, 0]
    tmp = DATA3.copy()
    gen_parcels = gen.parcels(parcelmap, labels=parcelseq)
    new_iterator = gen.data_generator(tmp, gen_parcels)
    for i, slice_ in enumerate(new_iterator):
        assert_equal((expected[i],), slice_[1].shape)


def test_sliceparcel():
    parcelmap = np.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
    parcelseq = ((1, 2), 0, 2)
    o = np.zeros(parcelmap.shape)
    iterator = gen.slice_parcels(parcelmap, labels=parcelseq)
    for i, pair in enumerate(iterator):
        a, s = pair
        o[a][s] = i
    assert_array_equal(o,
                       np.array([[1,1,1,0,2],
                                 [4,4,3,3,5],
                                 [7,7,7,7,8]]))
