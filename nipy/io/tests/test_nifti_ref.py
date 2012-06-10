# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test conversion between NIFTI and NIPY conventions.  The algorithms are
mostly written out in the :mod:`nipy.io.nifti_ref` docstrings.
"""
import warnings

from copy import copy

import numpy as np

import nibabel as nib
from nibabel.affines import from_matvec

from ...core.api import (Image,
                         AffineTransform as AT,
                         CoordinateSystem as CS)
from ...core.reference.spaces import (unknown_csm, scanner_csm, aligned_csm,
                                      talairach_csm, mni_csm, vox2mni)

from ..files import load
from ..nifti_ref import (nipy2nifti, nifti2nipy, NiftiError)

from nose.tools import assert_equal, assert_true, assert_false, assert_raises
from numpy.testing import assert_almost_equal, assert_array_equal

from ...testing import anatfile, funcfile

def copy_of(fname):
    # Make a fresh copy of a image stored in a file
    img = load(fname)
    hdr = img.metadata['header'].copy()
    return Image(img.get_data().copy(),
                 copy(img.coordmap),
                 {'header': hdr})


def setup():
    # Suppress warnings during tests
    warnings.simplefilter("ignore")


def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


def test_basic_nipy2nifti():
    # Go from nipy image to header and data for nifti
    fimg = copy_of(funcfile)
    hdr = fimg.metadata['header']
    data = fimg.get_data()
    # Header is preserved
    # Put in some information to check header is preserved
    hdr['slice_duration'] = 0.25
    ni_img = nipy2nifti(fimg)
    new_hdr = ni_img.get_header()
    # header copied on the way through
    assert_false(hdr is new_hdr)
    # Check information preserved
    assert_equal(hdr['slice_duration'], new_hdr['slice_duration'])
    assert_array_equal(data, ni_img.get_data())
    # Shape obviously should be same
    assert_equal(ni_img.shape, fimg.shape)


def test_xyz_affines():
    fimg = copy_of(funcfile)
    data = fimg.get_data()
    # Check conversion to xyz affable
    # Roll time to front in array
    fimg_t0 = fimg.reordered_axes((3, 0, 1, 2))
    # Nifti conversion rolls it back
    assert_array_equal(nipy2nifti(fimg_t0).get_data(), data)
    # Roll time to position 1
    fimg_t0 = fimg.reordered_axes((0, 3, 1, 2))
    assert_array_equal(nipy2nifti(fimg_t0).get_data(), data)
    # Check bad names cause NiftiError
    out_coords = fimg.reference.coord_names
    bad_img = fimg.renamed_reference(**{out_coords[0]: 'not a known axis'})
    assert_raises(NiftiError, nipy2nifti, bad_img)
    # Check xyz works for not strict
    bad_img = fimg.renamed_reference(**dict(zip(out_coords, 'xyz')))
    assert_array_equal(nipy2nifti(bad_img, strict=False).get_data(), data)
    # But fails for strict
    assert_raises(NiftiError, nipy2nifti, bad_img, strict=True)
    # 3D is OK
    aimg = copy_of(anatfile)
    adata = aimg.get_data()
    assert_array_equal(nipy2nifti(aimg).get_data(), adata)
    # For now, always error on 2D (this depends on as_xyz_image)
    assert_raises(NiftiError, nipy2nifti, aimg[:, :, 1])
    assert_raises(NiftiError, nipy2nifti, aimg[:, 1, :])
    assert_raises(NiftiError, nipy2nifti, aimg[1, :, :])
    # Do not allow spaces not in the NIFTI canon
    unknown_cs = unknown_csm(3)
    displaced_img = fimg.renamed_reference(
        **dict(zip(out_coords[:3], unknown_cs.coord_names)))
    assert_raises(NiftiError, nipy2nifti, displaced_img)


def test_orthogonal_dims():
    # Test whether conversion to nifti raises an error for non-orthogonal
    # non-spatial dimensions
    # This affine is all nicely diagonal
    aff = from_matvec(np.diag([2., 3, 4, 5, 6]), [10, 11, 12, 13, 14])
    data = np.random.normal(size=(3, 4, 5, 6, 7))
    img = Image(data, vox2mni(aff))
    def as3d(aff):
        return from_matvec(aff[:3, :3], aff[:3, -1])
    assert_array_equal(nipy2nifti(img).get_affine(), as3d(aff))
    # Non-orthogonal spatial dimensions OK
    aff[:3, :3] = np.random.normal(size=(3, 3))
    img = Image(data, vox2mni(aff))
    assert_array_equal(nipy2nifti(img).get_affine(), as3d(aff))
    # Space must be orthogonal to time etc
    aff[0, 3] = 0.1
    assert_raises(NiftiError, nipy2nifti, img)
    aff[0, 3] = 0
    assert_array_equal(nipy2nifti(img).get_affine(), as3d(aff))
    aff[3, 0] = 0.1
    assert_raises(NiftiError, nipy2nifti, img)
    aff[3, 0] = 0
    assert_array_equal(nipy2nifti(img).get_affine(), as3d(aff))
    aff[4, 0] = 0.1
    assert_raises(NiftiError, nipy2nifti, img)


def test_dim_info():
    # Test slice, freq, phase get set OK
    fimg = copy_of(funcfile)
    hdr = fimg.metadata['header']
    assert_equal(hdr.get_dim_info(), (None, None, None))
    ni_img = nipy2nifti(fimg)
    assert_equal(ni_img.get_header().get_dim_info(), (None, None, None))
    data = fimg.get_data()
    cmap = fimg.coordmap
    for i in range(3):
        for order, name in enumerate(('freq', 'phase', 'slice')):
            ncmap = cmap.renamed_domain({i: name})
            ni_img = nipy2nifti(Image(data, ncmap, {'header': hdr}))
            exp_info = [None, None, None]
            exp_info[order] = i
            assert_equal(ni_img.get_header().get_dim_info(),
                         tuple(exp_info))
    ncmap = cmap.renamed_domain(
        dict(zip(range(3), ('phase', 'slice', 'freq'))))
    ni_img = nipy2nifti(Image(data, ncmap, {'header': hdr}))
    assert_equal(ni_img.get_header().get_dim_info(), (2, 0, 1))


def test_time_like_matching():
    # Check checks for matching time-like axes
    shape = (2, 3, 4, 5, 6)
    shape_shifted = (2, 3, 4, 6, 5)
    data = np.random.normal(size=shape)
    aff = np.diag([3, 4, 5, 6, 7, 1])
    mni_names = mni_csm(3).coord_names
    time_cans = ('t', 'hz', 'ppm', 'rads')
    aliases = dict(t='time',
                   hz='frequency-hz',
                   ppm='concentration-ppm',
                   rads='radians/s')
    all_names = set(time_cans + tuple(v for v in aliases.values()))
    for time_like in time_cans:
        alias = aliases[time_like]
        for name in (time_like, alias):
            # Names match
            cmap = AT(CS(('i', 'j', 'k', name, 'u')),
                    CS(mni_names + (name, 'u')), aff)
            assert_equal(nipy2nifti(Image(data, cmap)).shape, shape)
            cmap = AT(CS(('i', 'j', 'k', 'u', name)),
                    CS(mni_names + ('u', name)), aff)
            assert_equal(nipy2nifti(Image(data, cmap)).shape, shape_shifted)
            # No time-like in output is OK
            cmap = AT(CS(('i', 'j', 'k', 'u', name)),
                    CS(mni_names + ('u', 'v')), aff)
            assert_equal(nipy2nifti(Image(data, cmap)).shape, shape_shifted)
            # No time-like in input is OK
            cmap = AT(CS(('i', 'j', 'k', 'u', 'v')),
                    CS(mni_names + ('u', name)), aff)
            assert_equal(nipy2nifti(Image(data, cmap)).shape, shape_shifted)
            # Time-like in both, but not matching, not OK
            cmap = AT(CS(('i', 'j', 'k', 'u', name)),
                    CS(mni_names + (name, 'u')), aff)
            assert_raises(NiftiError, nipy2nifti, Image(data, cmap))
            # Time like in both with no match between but no match elsewhere
            # Actually this does cause a problem for non-zero time offset and
            # time axes, but we test that elsewhere.
            cmap = AT(CS(('i', 'j', 'k', 'u', name)),
                    CS(mni_names + ('u', name)),
                    np.diag([3, 4, 5, 6, 0, 1]))
            assert_equal(nipy2nifti(Image(data, cmap)).shape, shape_shifted)
            cmap = AT(CS(('i', 'j', 'k', 'u', name)),
                    CS(mni_names + (name, 'u')),
                    np.diag([3, 4, 5, 0, 0, 1]))
            assert_equal(nipy2nifti(Image(data, cmap)).shape, shape_shifted)
        # Matching to own alias is OK
        cmap = AT(CS(('i', 'j', 'k', time_like, 'u')),
                CS(mni_names + (alias, 'u')), aff)
        assert_equal(nipy2nifti(Image(data, cmap)).shape, shape)
        cmap = AT(CS(('i', 'j', 'k', alias, 'u')),
                CS(mni_names + (time_like, 'u')), aff)
        assert_equal(nipy2nifti(Image(data, cmap)).shape, shape)
        # But not to another time-like name
        others = all_names.difference((time_like, alias))
        for name in others:
            cmap = AT(CS(('i', 'j', 'k', time_like, 'u')),
                    CS(mni_names + (name, 'u')), aff)
            assert_raises(NiftiError, nipy2nifti, Image(data, cmap))
            cmap = AT(CS(('i', 'j', 'k', name, 'u')),
                    CS(mni_names + (time_like, 'u')), aff)
            assert_raises(NiftiError, nipy2nifti, Image(data, cmap))
        # It's OK to have more than one time-like, but the order of recognition
        # is 't', 'hz', 'ppm', 'rads'
        for i, better in enumerate(time_cans[:-1]):
            for worse in time_cans[i+1:]:
                cmap = AT(CS(('i', 'j', 'k', better, worse)),
                        CS(mni_names + (better, worse)), aff)
                assert_equal(nipy2nifti(Image(data, cmap)).shape, shape)
                cmap = AT(CS(('i', 'j', 'k', worse, better)),
                        CS(mni_names + (worse, better)), aff)
                assert_equal(nipy2nifti(Image(data, cmap)).shape, shape_shifted)
                # Even if better is only in output
                cmap = AT(CS(('i', 'j', 'k', worse, 'u')),
                        CS(mni_names + (worse, better)), aff)
                assert_equal(nipy2nifti(Image(data, cmap)).shape, shape_shifted)


def test_time_pixdims():
    # Pixdims get moved across when a no-time extra axis is added
    shape = (2, 3, 4, 5, 6, 7)
    data = np.random.normal(size=shape)
    aff = np.diag([3, 4, 5, 6, 7, 8, 1])
    mni_names = mni_csm(3).coord_names
    in_cs = CS('ikjlmn')
    cmap = AT(in_cs, CS(mni_names + tuple('tuv')), aff)
    hdr = nipy2nifti(Image(data, cmap)).get_header()
    assert_equal(hdr.get_zooms(), (3, 4, 5, 6, 7, 8))
    cmap = AT(in_cs, CS(mni_names + tuple('quv')), aff)
    hdr = nipy2nifti(Image(data, cmap)).get_header()
    assert_equal(hdr.get_zooms(), (3, 4, 5, 0, 6, 7, 8))


def test_xyzt_units():
    # Whether xyzt_unit field gets set correctly
    fimg_orig = copy_of(funcfile)
    # Put time in output, input and both
    data = fimg_orig.get_data()
    hdr = fimg_orig.metadata['header']
    aff = fimg_orig.coordmap.affine
    out_names = fimg_orig.reference.coord_names
    # Time in input only
    cmap_t_in = AT('ijkt', out_names[:3] + ('q',), aff)
    img_t_in = Image(data, cmap_t_in, {'header': hdr.copy()})
    # Time in output only
    cmap_t_out = AT('ijkl', out_names[:3] + ('t',), aff)
    img_t_out = Image(data, cmap_t_out, {'header': hdr.copy()})
    # Time in both
    cmap_t_b = AT('ijkt', out_names[:3] + ('t',), aff)
    img_t_b = Image(data, cmap_t_b, {'header': hdr.copy()})
    # In neither
    cmap_t_no = AT('ijkl', out_names[:3] + ('q',), aff)
    img_t_no = Image(data, cmap_t_no, {'header': hdr.copy()})
    # Check the default
    assert_equal(hdr.get_xyzt_units(), ('mm', 'sec'))
    # That default survives nifti conversion
    for img in (img_t_in, img_t_out, img_t_b):
        ni_img = nipy2nifti(img)
        assert_equal(ni_img.get_header().get_xyzt_units(), ('mm', 'sec'))
    # Now with no time
    for img in (img_t_no, img_t_b[...,0]):
        ni_img = nipy2nifti(img)
        assert_equal(ni_img.get_header().get_xyzt_units(), ('mm', 'unknown'))
    # Change to other time-like
    for units, name0, name1 in (('sec', 't', 'time'),
                                ('hz', 'hz', 'frequency-hz'),
                                ('ppm', 'ppm', 'concentration-ppm'),
                                ('rads', 'rads', 'radians/s')):
        for name in (name0, name1):
            new_img = img_t_out.renamed_reference(t=name)
            ni_img = nipy2nifti(new_img)
            assert_equal(ni_img.get_header().get_xyzt_units(), ('mm', units))
            new_img = img_t_in.renamed_axes(t=name)
            ni_img = nipy2nifti(new_img)
            assert_equal(ni_img.get_header().get_xyzt_units(), ('mm', units))
            new_img = img_t_b.renamed_axes(t=name).renamed_reference(t=name)
            ni_img = nipy2nifti(new_img)
            assert_equal(ni_img.get_header().get_xyzt_units(), ('mm', units))


def test_time_axes_4th():
    # Check time-like axes rolled to be 4th, and pixdims match
    data = np.random.normal(size=(2, 3, 4, 5, 6, 7))
    aff = np.diag([2., 3, 4, 5, 6, 7, 1])
    xyz_names = talairach_csm(3).coord_names
    in_cs = CS('ijklmn')
    for time_like in ('t', 'hz', 'ppm', 'rads'):
        cmap = AT(in_cs, CS(xyz_names + (time_like, 'q', 'r')), aff)
        img = Image(data, cmap)
        # Time-like in correct position
        ni_img = nipy2nifti(img)
        assert_array_equal(ni_img.get_data(), data)
        assert_array_equal(ni_img.get_header().get_zooms(), (2, 3, 4, 5, 6, 7))
        # Time-like needs reordering
        cmap = AT(in_cs, CS(xyz_names + ('q', time_like, 'r')), aff)
        ni_img = nipy2nifti(Image(data, cmap))
        assert_array_equal(ni_img.get_data(), np.rollaxis(data, 4, 3))
        assert_array_equal(ni_img.get_header().get_zooms(), (2, 3, 4, 6, 5, 7))
        # And again
        cmap = AT(in_cs, CS(xyz_names + ('q', 'r', time_like)), aff)
        ni_img = nipy2nifti(Image(data, cmap))
        assert_array_equal(ni_img.get_data(), np.rollaxis(data, 5, 3))
        assert_array_equal(ni_img.get_header().get_zooms(), (2, 3, 4, 7, 5, 6))


def test_save_toffset():
    # Check toffset only gets set for time
    shape = (2, 3, 4, 5, 6, 7)
    data = np.random.normal(size = shape)
    aff = from_matvec(np.diag([2., 3, 4, 5, 6, 7]),
                              [11, 12, 13, 14, 15, 16])
    xyz_names = talairach_csm(3).coord_names
    in_cs = CS('ijklmn')
    for t_name in 't', 'time':
        cmap = AT(in_cs, CS(xyz_names + (t_name, 'q', 'r')), aff)
        ni_img = nipy2nifti(Image(data, cmap))
        assert_equal(ni_img.get_header()['toffset'], 14)
    for time_like in ('hz', 'ppm', 'rads'):
        cmap = AT(in_cs, CS(xyz_names + (time_like, 'q', 'r')), aff)
        ni_img = nipy2nifti(Image(data, cmap))
        assert_equal(ni_img.get_header()['toffset'], 0)
    # Check that non-matching time causes a nifti error when toffset !=0
    shape_shifted = (2, 3, 4, 6, 5, 7)
    for t_name in 't', 'time':
        # No toffset, this is OK
        cmap = AT(CS(('i', 'j', 'k', 'u', t_name, 'v')),
                  CS(xyz_names + ('u', t_name, 'v')),
                  np.diag([3, 4, 5, 6, 0, 7, 1]))
        assert_equal(nipy2nifti(Image(data, cmap)).shape, shape_shifted)
        # toffset, non-matching error
        aff_z1 = from_matvec(np.diag([2., 3, 4, 5, 0, 7]),
                             [11, 12, 13, 14, 15, 16])
        cmap = AT(CS(('i', 'j', 'k', 'u', t_name, 'v')),
                  CS(xyz_names + ('u', t_name, 'v')),
                  aff_z1)
        assert_raises(NiftiError, nipy2nifti, Image(data, cmap))
        # Unless fix0 set
        assert_equal(nipy2nifti(Image(data, cmap), fix0=True).shape,
                     shape_shifted)
        # Even this doesn't work if there is more than one zero row and column
        aff_z2 = from_matvec(np.diag([2., 3, 4, 0, 0, 7]),
                             [11, 12, 13, 14, 15, 16])
        cmap = AT(CS(('i', 'j', 'k', 'u', t_name, 'v')),
                  CS(xyz_names + ('u', t_name, 'v')),
                  aff_z2)
        assert_raises(NiftiError, nipy2nifti, Image(data, cmap), fix0=True)
    # No problem for non-time
    for t_name in 'hz', 'ppm', 'rads':
        cmap = AT(CS(('i', 'j', 'k', 'u', t_name, 'v')),
                  CS(xyz_names + ('u', t_name, 'v')),
                  aff)
        assert_equal(nipy2nifti(Image(data, cmap), fix0=True).shape,
                     shape_shifted)


def test_too_many_dims():
    data0 = np.zeros(range(2, 9))
    xyz_names = talairach_csm(3).coord_names
    cmap = AT(CS('ijktuvw'), CS(xyz_names + tuple('tuvw')), np.eye(8))
    assert_equal(nipy2nifti(Image(data0, cmap)).shape, tuple(range(2, 9)))
    # Too many dimensions
    data1 = np.zeros(range(2, 10))
    cmap = AT(CS('ijktuvwq'), CS(xyz_names + tuple('tuvwq')), np.eye(9))
    assert_raises(NiftiError, nipy2nifti, Image(data1, cmap))
    # No time adds a dimension
    cmap = AT(CS('ijkpuvw'), CS(xyz_names + tuple('puvw')), np.eye(8))
    assert_raises(NiftiError, nipy2nifti, Image(data0, cmap))


def test_no_time():
    # Check that no time axis results in extra length 1 dimension
    data = np.random.normal(size=(2, 3, 4, 5, 6, 7))
    aff = np.diag([2., 3, 4, 5, 6, 7, 1])
    xyz_names = talairach_csm(3).coord_names
    in_cs = CS('ijklmn')
    # No change in shape if there's a time-like axis
    for time_like in ('t', 'hz', 'ppm', 'rads'):
        cmap = AT(in_cs, CS(xyz_names + (time_like, 'q', 'r')), aff)
        ni_img = nipy2nifti(Image(data, cmap))
        assert_array_equal(ni_img.get_data(), data)
    # But there is if no time-like
    for no_time in ('random', 'words', 'I', 'thought', 'of'):
        cmap = AT(in_cs, CS(xyz_names + (no_time, 'q', 'r')), aff)
        ni_img = nipy2nifti(Image(data, cmap))
        assert_array_equal(ni_img.get_data(), data[:, :, :, None, :, :])


def test_save_spaces():
    # Test that intended output spaces get set into nifti
    data = np.random.normal(size=(2, 3, 4))
    aff = np.diag([2., 3, 4, 1])
    in_cs = CS('ijk')
    for label, csm in (('scanner', scanner_csm),
                       ('aligned', aligned_csm),
                       ('talairach', talairach_csm),
                       ('mni', mni_csm)):
        img = Image(data, AT(in_cs, csm(3), aff))
        ni_img = nipy2nifti(img)
        assert_equal(ni_img.get_header().get_value_label('sform_code'),
                     label)


def test_basic_load():
    # Just basic load
    data = np.random.normal(size=(2, 3, 4, 5))
    aff = np.diag([2., 3, 4, 1])
    ni_img = nib.Nifti1Image(data, aff)
    img = nifti2nipy(ni_img)
    assert_array_equal(img.get_data(), data)


def test_expand_to_3d():
    # Test 1D and 2D niftis
    # 1D and 2D with full sform or qform affines raise a NiftiError, because we
    # can't be sure which axes the affine refers to.  Should the image have 1
    # length axes prepended?  Or appended?
    xyz_aff = np.diag([2, 3, 4, 1])
    for size in (10,), (10, 2):
        data = np.random.normal(size=size)
        ni_img = nib.Nifti1Image(data, xyz_aff)
        # Default is aligned
        assert_raises(NiftiError, nifti2nipy, ni_img)
        hdr = ni_img.get_header()
        # The pixdim affine
        for label in 'scanner', 'aligned', 'talairach', 'mni':
            hdr.set_sform(xyz_aff, label)
            assert_raises(NiftiError, nifti2nipy, ni_img)
            hdr.set_sform(None)
            assert_raises(NiftiError, nifti2nipy, ni_img)
            hdr.set_sform(xyz_aff, label)
            assert_raises(NiftiError, nifti2nipy, ni_img)
            hdr.set_qform(None)


def test_load_cmaps():
    data = np.random.normal(size=range(7))
    xyz_aff = np.diag([2, 3, 4, 1])
    # Default with time-like
    ni_img = nib.Nifti1Image(data, xyz_aff)
    img = nifti2nipy(ni_img)
    exp_cmap = AT(CS('ijktuvw', name='voxels'),
                  aligned_csm(7),
                  np.diag([2, 3, 4, 1, 1, 1, 1, 1]))
    assert_equal(img.coordmap, exp_cmap)
    # xyzt_units sets time axis name
    hdr = ni_img.get_header()
    xyz_names = aligned_csm(3).coord_names
    full_aff = exp_cmap.affine
    reduced_data = data[:, :, :, 1:2, ...]
    for t_like, units, scaling in (
        ('t', 'sec', 1),
        ('t', 'msec', 1/1000.),
        ('t', 'usec', 1/1000000.),
        ('hz', 'hz', 1),
        ('ppm', 'ppm', 1),
        ('rads', 'rads', 1)):
        hdr.set_xyzt_units('mm', units)
        img = nifti2nipy(ni_img)
        in_cs = CS(('i', 'j', 'k', t_like, 'u', 'v', 'w'), name='voxels')
        out_cs = CS(xyz_names + (t_like, 'u', 'v', 'w'), name='aligned')
        if scaling == 1:
            exp_aff = full_aff
        else:
            diag = np.ones((8,))
            diag[3] = scaling
            exp_aff = np.dot(np.diag(diag), full_aff)
        exp_cmap = AT(in_cs, out_cs, exp_aff)
        assert_equal(img.coordmap, exp_cmap)
        assert_array_equal(img.get_data(), data)
        # Even if the image axis length is 1, we keep out time dimension, if
        # there is specific scaling implying time-like
        ni_img_t = nib.Nifti1Image(reduced_data, xyz_aff, hdr)
        img = nifti2nipy(ni_img_t)
        assert_equal(img.coordmap, exp_cmap)
        assert_array_equal(img.get_data(), reduced_data)


def test_load_no_time():
    # Without setting anything else, length 1 at position 3 makes time go away
    ns_dims = (5, 6, 7)
    xyz_aff = np.diag([2, 3, 4, 1])
    xyz_names = aligned_csm(3).coord_names[:3]
    in_names = tuple('ijkuvw')
    out_names = xyz_names + tuple('uvw')
    for n_ns in 1, 2, 3:
        ndim = 3 + n_ns
        data = np.random.normal(size=(2, 3, 4, 1) + ns_dims[:n_ns])
        ni_img_no_t = nib.Nifti1Image(data, xyz_aff)
        cmap_no_t = AT(CS(in_names[:ndim], name='voxels'),
                       CS(out_names[:ndim], name='aligned'),
                       np.diag([2, 3, 4] + [1] * n_ns + [1]))
        img = nifti2nipy(ni_img_no_t)
        assert_equal(img.coordmap, cmap_no_t)
    # We add do time if 4th axis of length 1 is the last axis
    data41 = np.zeros((3, 4, 5, 1))
    ni_img_41 = nib.Nifti1Image(data41, xyz_aff)
    cmap_41 = AT(CS('ijkt', name='voxels'),
                   CS(xyz_names + ('t',), name='aligned'),
                   np.diag([2, 3, 4, 1, 1]))
    img = nifti2nipy(ni_img_41)
    assert_equal(img.coordmap, cmap_41)


def test_load_toffset():
    # Test toffset gets set into affine only for time
    data = np.random.normal(size=range(5))
    xyz_aff = np.diag([2, 3, 4, 1])
    # Default with time-like and no toffset
    ni_img = nib.Nifti1Image(data, xyz_aff)
    hdr = ni_img.get_header()
    img = nifti2nipy(ni_img)
    exp_aff = np.diag([2., 3, 4, 1, 1, 1])
    in_cs = CS('ijktu', name='voxels')
    xyz_names = aligned_csm(3).coord_names
    out_cs = CS(xyz_names + tuple('tu'), name='aligned')
    assert_equal(hdr['toffset'], 0)
    assert_equal(img.coordmap, AT(in_cs, out_cs, exp_aff))
    # Set toffset and expect in affine
    hdr['toffset'] = 42
    exp_aff[3, -1] = 42
    assert_equal(nifti2nipy(ni_img).coordmap, AT(in_cs, out_cs, exp_aff))
    # Make time axis into hz and expect not to see toffset
    hdr.set_xyzt_units('mm', 'hz')
    in_cs_hz = CS(('i', 'j', 'k', 'hz', 'u'), name='voxels')
    out_cs_hz = CS(xyz_names + ('hz', 'u'), name='aligned')
    exp_aff[3, -1] = 0
    assert_equal(nifti2nipy(ni_img).coordmap, AT(in_cs_hz, out_cs_hz, exp_aff))


def test_load_spaces():
    # Test spaces get read correctly
    shape = np.array((6, 5, 4, 3, 2))
    zooms = np.array((2, 3, 4, 5, 6))
    data = np.random.normal(size=shape)
    # Default with no affine in header, or in image
    ni_img = nib.Nifti1Image(data, None)
    hdr = ni_img.get_header()
    hdr.set_zooms(zooms)
    # Expected affine is from the pixdims and the center of the image.  Default
    # is also flipped X.
    offsets = (1 - shape[:3]) / 2. * zooms[:3] * (-1, 1, 1)
    exp_aff = from_matvec(np.diag([-2, 3, 4, 5, 6]),
                          list(offsets) + [0, 0])
    in_cs = CS('ijktu', name='voxels')
    exp_cmap = AT(in_cs, unknown_csm(5), exp_aff)
    assert_equal(nifti2nipy(ni_img).coordmap, exp_cmap)
    an_aff = from_matvec(np.diag([1.1, 2.2, 3.3]), [10, 11, 12])
    exp_aff = from_matvec(np.diag([1.1, 2.2, 3.3, 5, 6]), [10, 11, 12, 0, 0])
    for label, csm in (('scanner', scanner_csm),
                       ('aligned', aligned_csm),
                       ('talairach', talairach_csm),
                       ('mni', mni_csm)):
        hdr.set_sform(an_aff, label)
        assert_equal(nifti2nipy(ni_img).coordmap, AT(in_cs, csm(5), exp_aff))


def test_mm_scaling():
    # Test the micron and meter scale the affine right
    data = np.random.normal(size=range(4))
    xyz_aff = from_matvec(np.diag([2, 3, 4]), [11, 12, 13])
    exp_aff = from_matvec(np.diag([2, 3, 4, 1]), [11, 12, 13, 0])
    in_cs = CS('ijkt', name='voxels')
    out_cs = aligned_csm(4)
    # No space scaling
    ni_img = nib.Nifti1Image(data, xyz_aff)
    hdr = ni_img.get_header()
    assert_equal(hdr.get_xyzt_units(), ('unknown', 'unknown'))
    assert_equal(nifti2nipy(ni_img).coordmap, AT(in_cs, out_cs, exp_aff))
    # mm is assumed
    hdr.set_xyzt_units('mm')
    assert_equal(nifti2nipy(ni_img).coordmap, AT(in_cs, out_cs, exp_aff))
    # microns !
    hdr.set_xyzt_units('micron')
    scaler = np.diag([1 / 1000., 1 / 1000., 1 / 1000., 1, 1])
    assert_equal(nifti2nipy(ni_img).coordmap,
                 AT(in_cs, out_cs, np.dot(scaler, exp_aff)))
    # mm again !  This test implicitly asserts that the nifti image affine is
    # not being changed by the conversion routine, otherwise we'd pick up the
    # microns scaling above.
    hdr.set_xyzt_units('mm')
    assert_equal(nifti2nipy(ni_img).coordmap, AT(in_cs, out_cs, exp_aff))
    # meters !
    hdr.set_xyzt_units('meter')
    scaler = np.diag([1000., 1000., 1000., 1, 1])
    assert_equal(nifti2nipy(ni_img).coordmap,
                 AT(in_cs, out_cs, np.dot(scaler, exp_aff)))


def test_load_dim_info():
    # Test freq, phase, slice get set correctly on load
    data = np.random.normal(size=range(3))
    xyz_aff = from_matvec(np.diag([2, 3, 4]), [11, 12, 13])
    in_cs = CS('ijk', name='voxels')
    out_cs = aligned_csm(3)
    # Just confirm that the default leads to no axis renaming
    ni_img = nib.Nifti1Image(data, xyz_aff)
    hdr = ni_img.get_header()
    assert_equal(hdr.get_dim_info(), (None, None, None))
    assert_equal(nifti2nipy(ni_img).coordmap, AT(in_cs, out_cs, xyz_aff))
    # But now...
    hdr.set_dim_info(freq=1)
    assert_equal(nifti2nipy(ni_img).coordmap,
                 AT(CS(('i', 'freq', 'k'), "voxels"), out_cs, xyz_aff))
    hdr.set_dim_info(freq=2)
    assert_equal(nifti2nipy(ni_img).coordmap,
                 AT(CS(('i', 'j', 'freq'), "voxels"), out_cs, xyz_aff))
    hdr.set_dim_info(phase=1)
    assert_equal(nifti2nipy(ni_img).coordmap,
                 AT(CS(('i', 'phase', 'k'), "voxels"), out_cs, xyz_aff))
    hdr.set_dim_info(slice=0)
    assert_equal(nifti2nipy(ni_img).coordmap,
                 AT(CS(('slice', 'j', 'k'), "voxels"), out_cs, xyz_aff))
    hdr.set_dim_info(freq=1, phase=0, slice=2)
    assert_equal(nifti2nipy(ni_img).coordmap,
                 AT(CS(('phase', 'freq', 'slice'), "voxels"), out_cs, xyz_aff))
