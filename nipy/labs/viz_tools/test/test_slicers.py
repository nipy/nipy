# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import pytest

try:
    import matplotlib as mpl
except ImportError:
    pytest.skip("Could not import matplotlib", allow_module_level=True)

from ..anat_cache import find_mni_template
from ..slicers import demo_ortho_slicer

################################################################################
# Some smoke testing for graphics-related code

def test_demo_ortho_slicer():
    # This is only a smoke test
    # conditioned on presence of MNI templated
    if not find_mni_template():
        pytest.skip("MNI Template is absent for the smoke test")
    # Make really sure that we don't try to open an Xserver connection.
    mpl.use('svg')
    import matplotlib.pyplot as plt
    plt.switch_backend('svg')
    demo_ortho_slicer()
