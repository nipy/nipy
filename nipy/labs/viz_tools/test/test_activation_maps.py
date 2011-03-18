# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from ..activation_maps import demo_plot_map

try:
    import matplotlib as mp
    # Make really sure that we don't try to open an Xserver connection.
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    do_test = True
except ImportError:
    do_test = False


if do_test:
    def test_demo_plot_map():
        # This is only a smoke test
        mp.use('svg', warn=False)
        import pylab as pl
        pl.switch_backend('svg')
        demo_plot_map()

