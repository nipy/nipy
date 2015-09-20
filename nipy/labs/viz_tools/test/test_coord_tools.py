from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from ..coord_tools import (coord_transform, find_cut_coords,
                           find_maxsep_cut_coords)

from numpy.testing import assert_array_equal

def test_coord_transform_trivial():
    sform = np.eye(4)
    x = np.random.random((10,))
    y = np.random.random((10,))
    z = np.random.random((10,))

    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x, x_)
    np.testing.assert_array_equal(y, y_)
    np.testing.assert_array_equal(z, z_)

    sform[:, -1] = 1
    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x+1, x_)
    np.testing.assert_array_equal(y+1, y_)
    np.testing.assert_array_equal(z+1, z_)


def test_find_cut_coords():
    map = np.zeros((100, 100, 100))
    x_map, y_map, z_map = 50, 10, 40
    map[x_map-30:x_map+30, y_map-3:y_map+3, z_map-10:z_map+10] = 1
    x, y, z = find_cut_coords(map, mask=np.ones(map.shape, np.bool))
    np.testing.assert_array_equal(
                        (int(round(x)), int(round(y)), int(round(z))),
                                (x_map, y_map, z_map))


def test_find_maxsep_cut_coords():
    # Test find_maxsep_cut_coords function
    assert_array_equal(
        find_maxsep_cut_coords(np.ones((2, 3, 5)), np.eye(4)), list(range(5)))
    assert_array_equal(
        find_maxsep_cut_coords(np.ones((2, 3, 5)), np.eye(4), threshold=1),
        list(range(5)))
    assert_array_equal(
        find_maxsep_cut_coords(np.ones((2, 3, 4)), np.eye(4), n_cuts=4),
        list(range(4)))
    map_3d = np.ones((2, 3, 5))
    map_3d[:, :, 1] = 0
    assert_array_equal(
        find_maxsep_cut_coords(map_3d, np.eye(4), n_cuts=4), [0, 2, 3, 4])
    map_3d[:, :, 1] = 0.5
    assert_array_equal(
        find_maxsep_cut_coords(map_3d, np.eye(4), n_cuts=4, threshold=0.6),
        [0, 2, 3, 4])
