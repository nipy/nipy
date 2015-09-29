from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np

from .... import load_image
from ....testing import anatfile
from ..histogram_registration import HistogramRegistration

from numpy.testing import assert_array_almost_equal

anat_img = load_image(anatfile)

def test_registers():
    # Test registration to self returns identity
    for cost, interp, affine_type in (('cc', 'pv', 'rigid'),
                                      ('cc', 'tri', 'rigid'),
                                      ('cc', 'rand', 'rigid'),
                                      ('cc', 'pv', 'similarity'),
                                      ('cc', 'pv', 'affine'),
                                      ('cr', 'pv', 'rigid'),
                                      ('cr', 'pv', 'rigid'),
                                      ('crl1', 'pv', 'rigid'),
                                      ('mi', 'pv', 'rigid'),
                                      ('nmi', 'pv', 'rigid'),
                                     ):
        R = HistogramRegistration(anat_img, anat_img,
                                  similarity=cost,
                                  interp=interp)
        R.subsample([2,2,2])
        affine = R.optimize(affine_type)
        yield assert_array_almost_equal, affine.as_affine(), np.eye(4), 2
