# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Smoke testing the cm module
"""

import pytest

try:
    import matplotlib as mpl
    # Make really sure that we don't try to open an Xserver connection.
    mpl.use('svg')
    import matplotlib.pyplot as plt
    plt.switch_backend('svg')
except ImportError:
    pytest.skip("Could not import matplotlib", allow_module_level=True)

from ..cm import dim_cmap, replace_inside


def test_dim_cmap():
    # This is only a smoke test
    mpl.use('svg')
    import matplotlib.pyplot as plt
    dim_cmap(plt.cm.jet)


def test_replace_inside():
    # This is only a smoke test
    mpl.use('svg')
    import matplotlib.pyplot as plt
    plt.switch_backend('svg')
    replace_inside(plt.cm.jet, plt.cm.hsv, .2, .8)
    # We also test with gnuplot, which is defined using function
    if hasattr(plt.cm, 'gnuplot'):
        # gnuplot is only in recent version of MPL
        replace_inside(plt.cm.gnuplot, plt.cm.gnuplot2, .2, .8)
