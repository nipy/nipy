# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Smoke testing the cm module
"""

from ..cm import dim_cmap, replace_inside

from ....utils.optpkg import optional_package
pl, have_mpl, setup_module = optional_package('pylab')


if have_mpl:
    import matplotlib as mp
    # Make really sure that we don't try to open an Xserver connection.
    mp.use('svg', warn=False)
    pl.switch_backend('svg')


def test_dim_cmap():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    dim_cmap(pl.cm.jet)


def test_replace_inside():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    replace_inside(pl.cm.jet, pl.cm.hsv, .2, .8)

