# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import warnings
import numpy as np

import nibabel as nib
from nibabel.affines import append_diag

from ...core.api import (CoordinateMap, AffineTransform, CoordinateSystem,
                           lps_output_coordnames, ras_output_coordnames)
from ...core.reference.spaces import (unknown_csm, scanner_csm, aligned_csm,
                                      talairach_csm)

from ..nifti_ref import (ni_affine_pixdim_from_affine,
                         get_input_cs, get_output_cs)

from nose.tools import assert_equal, assert_true, assert_false, assert_raises
from numpy.testing import assert_almost_equal


shape = range(1,8)
step = np.arange(1,8)

output_axes = lps_output_coordnames + tuple('tuvw')
input_axes = 'ijktuvw'
lps = lps_output_coordnames # shorthand

def setup():
    # Suppress warnings during tests
    warnings.simplefilter("ignore")


def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


def test_ni_pix_from_aff():
    # Test ni_pixdim_from_affine
    # A 3D affine
    aff = np.random.standard_normal((4,4))
    aff[-1] = [0,0,0,1]
    function_domain = CoordinateSystem('ijk', 'input')
    function_range = CoordinateSystem(lps, 'output')
    threed = AffineTransform(function_domain, function_range, aff)
    # Corresponding 4D
    aff4 = append_diag(aff, [3.5])
    fourd = AffineTransform('ijkl', lps_output_coordnames + ('t',), aff4)
    yield assert_almost_equal, aff, threed.affine
    yield assert_almost_equal, fourd.affine[3:,3:], np.diag([3.5,1])

    # get the pixdim back out
    A, p = ni_affine_pixdim_from_affine(fourd)
    yield assert_almost_equal, aff, A.affine
    yield assert_almost_equal, p, 3.5

    # try strict
    A, p = ni_affine_pixdim_from_affine(fourd, strict=True)

    # try using RAS
    cmap = fourd.renamed_range(dict(zip(lps, ras_output_coordnames)))
    A, p = ni_affine_pixdim_from_affine(cmap, strict=True)

    # will have been flipped to LPS
    yield assert_almost_equal, A.affine, np.dot(np.diag([-1,-1,1,1]),aff)
    yield assert_equal, A.function_range.coord_names, lps

    # use coordinates that aren't OK and strict raises an exception
    cmap = fourd.renamed_range(dict(zip(lps, 'xyz')))
    yield assert_raises, ValueError, ni_affine_pixdim_from_affine, cmap, \
        True

    # use coordinates that aren't OK and not strict just guesses LPS
    cmap4 = fourd.renamed_range(dict(zip(lps, 'xyz')))
    A, p =  ni_affine_pixdim_from_affine(cmap4, False)
    yield assert_almost_equal, A.affine, aff
    yield assert_equal, A.function_range.coord_names, lps
    yield assert_almost_equal, p, 3.5

    # non-square affine fails
    Z = np.random.standard_normal((5,4))
    Z[-1] = [0,0,0,1]
    affine = AffineTransform.from_params('ijk', 'xyzt', Z)
    yield assert_raises, ValueError, ni_affine_pixdim_from_affine, affine

    # CoordinateMap fails
    ijk = CoordinateSystem('ijk')
    xyz = CoordinateSystem('xzy')
    cmap = CoordinateMap(ijk, xyz, np.exp)
    yield assert_raises, ValueError, ni_affine_pixdim_from_affine, cmap, True
    # non-diagonal above 3rd dimension, with strict True raises an exception
    cmap5 = cmap4.renamed_range(dict(zip('xyz', lps)))
    cmap5.affine[3,-1] = 4.
    yield assert_raises, ValueError, ni_affine_pixdim_from_affine, cmap5, True
    B, p = ni_affine_pixdim_from_affine(cmap5)
    yield assert_equal, p, 3.5


def test_input_cs():
    # Test ability to detect input coordinate system
    # I believe nifti is the only format to specify interesting meanings for the
    # input axes
    for hdr in (nib.Spm2AnalyzeHeader(), nib.Nifti1Header()):
        for shape, names in (((2,), 'i'),
                            ((2,3), 'ij'),
                            ((2,3,4), 'ijk'),
                            ((2,3,4,5), 'ijkl')):
            hdr.set_data_shape(shape)
            assert_equal(get_input_cs(hdr), CoordinateSystem(names, 'voxel'))
    hdr = nib.Nifti1Header()
    # Just confirm that the default leads to no axis renaming
    hdr.set_data_shape((2,3,4))
    hdr.set_dim_info(None, None, None) # the default
    assert_equal(get_input_cs(hdr), CoordinateSystem('ijk', 'voxel'))
    # But now...
    hdr.set_dim_info(freq=1)
    assert_equal(get_input_cs(hdr),
                 CoordinateSystem(('i', 'freq', 'k'), "voxel"))
    hdr.set_dim_info(freq=2)
    assert_equal(get_input_cs(hdr),
                 CoordinateSystem(('i', 'j', 'freq'), "voxel"))
    hdr.set_dim_info(phase=1)
    assert_equal(get_input_cs(hdr),
                 CoordinateSystem(('i', 'phase', 'k'), "voxel"))
    hdr.set_dim_info(freq=1, phase=0, slice=2)
    assert_equal(get_input_cs(hdr),
                 CoordinateSystem(('phase', 'freq', 'slice'), "voxel"))


def test_output_cs():
    # Test return of output coordinate system from header
    # With our current use of nibabel, there is always an xyz output.  But, with
    # nifti, the xform codes can specify one of four known output spaces.
    # But first - length is always 3 until we have more than 3 input dimensions
    cs = unknown_csm(3) # A length 3 xyz output
    hdr = nib.Nifti1Header()
    hdr.set_data_shape((2,))
    assert_equal(get_output_cs(hdr), cs)
    hdr.set_data_shape((2,3))
    assert_equal(get_output_cs(hdr), cs)
    hdr.set_data_shape((2,3,4))
    assert_equal(get_output_cs(hdr), cs)
    # With more than 3 inputs, the output dimensions expand
    hdr.set_data_shape((2,3,4,5))
    assert_equal(get_output_cs(hdr), unknown_csm(4))
    # Now, nifti can change the output labels with xform codes
    hdr['qform_code'] = 1
    assert_equal(get_output_cs(hdr), scanner_csm(4))
    hdr['qform_code'] = 3
    assert_equal(get_output_cs(hdr), talairach_csm(4))
    hdr['sform_code'] = 2
    assert_equal(get_output_cs(hdr), aligned_csm(4))
    hdr['sform_code'] = 0
    assert_equal(get_output_cs(hdr), talairach_csm(4))
