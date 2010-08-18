# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing registration

"""

from nipy.io.imageformats import load
from nipy.testing import anatfile
from nipy.neurospin.registration import register


anat_img = load(anatfile)


def test_registration():
    aff = register(anat_img, anat_img)
    print aff


        
