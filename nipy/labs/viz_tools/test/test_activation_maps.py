# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import tempfile

import numpy as np

from nose import SkipTest
try:
    import matplotlib as mp
    # Make really sure that we don't try to open an Xserver connection.
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
except ImportError:
    raise SkipTest('Could not import matplotlib')

from ..activation_maps import demo_plot_map, plot_anat, plot_map
from ..anat_cache import mni_sform, _AnatCache



def test_demo_plot_map():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    demo_plot_map()
    # Test the black background code path
    demo_plot_map(black_bg=True)


def test_plot_anat():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    data = np.zeros((20, 20, 20))
    data[3:-3, 3:-3, 3:-3] = 1
    ortho_slicer = plot_anat(data, mni_sform, dim=True)
    ortho_slicer = plot_anat(data, mni_sform, cut_coords=(80, -120, -60))
    # Saving forces a draw, and thus smoke-tests the axes locators
    pl.savefig(tempfile.TemporaryFile())
    ortho_slicer.edge_map(data, mni_sform, color='c')

    # Test saving with empty plot
    z_slicer = plot_anat(anat=False, slicer='z')
    pl.savefig(tempfile.TemporaryFile())
    z_slicer = plot_anat(slicer='z')
    pl.savefig(tempfile.TemporaryFile())
    z_slicer.edge_map(data, mni_sform, color='c')
    # Smoke test coordinate finder, with and without mask
    plot_map(np.ma.masked_equal(data, 0), mni_sform, slicer='x')
    plot_map(data, mni_sform, slicer='y')


def test_anat_cache():
    # A smoke test, that can work only if the templates are installed
    try:
        _AnatCache.get_blurred()
    except OSError:
        "The templates are not there"
        pass


def test_plot_map_empty():
    # Test that things don't crash when we give a map with nothing above
    # threshold
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    data = np.zeros((20, 20, 20))
    plot_anat(data, mni_sform)
    plot_map(data, mni_sform, slicer='y', threshold=1)
    pl.close('all')


def test_plot_map_with_auto_cut_coords():
    import pylab as pl
    pl.switch_backend('svg')
    data = np.zeros((20, 20, 20))
    data[3:-3, 3:-3, 3:-3] = 1

    for slicer in 'xyz':
        plot_map(data, np.eye(4), cut_coords=None, slicer=slicer,
                 black_bg=True)
