# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os

import numpy as np

from nibabel.tmpdirs import TemporaryDirectory
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


def test_plot_anat():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    ortho_slicer = plot_anat()
    data = np.zeros((100, 100, 100))
    data[3:-3, 3:-3, 3:-3] = 1
    ortho_slicer.edge_map(data, mni_sform, color='c')


def test_plot_map_memmap():
    """ Test that memmap arrays, as with arrays returned by nibabel, don't
        cause crashes.
    """
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    data = np.zeros((10, 10, 10))
    data[2:5, 2:5, 2:5] = np.random.random(size=(3, 3, 3))
    with TemporaryDirectory() as dir_name:
        file_name = os.path.join(dir_name, 'tmp.npy')
        # Save and load to have a memmap file
        np.save(file_name, data)
        data = np.load(file_name, mmap_mode='r')
        affine = np.eye(4)
        plot_map(data, affine, threshold=.5)
        del data


def test_anat_cache():
    # A smoke test, that can work only if the templates are installed
    try:
        _AnatCache.get_blurred()
    except OSError:
        "The templates are not there"
        pass


