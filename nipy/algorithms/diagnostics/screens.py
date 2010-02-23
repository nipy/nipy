''' Diagnostic 4d image screen '''

import numpy as np

from ...core.api import Image, drop_io_dim, append_io_dim
from ...modalities.fmri import pca
from .timediff import time_slice_diffs


def screen(img4d, ncomp=10):
    ''' Diagnostic screen for 4d FMRI image

    Includes PCA, tsdiffana and mean, std, min, max images.

    Parameters
    ----------
    img4d : ``Image``
       4d image file
    ncomp : int, optional
       number of component images to return.  Default is 10

    Returns
    -------
    screen : dict
       with keys:

       * mean : mean image (all summaries are over last dimension)
       * std : standard deviation image
       * max : image of max
       * min : min
       * pca_res : dict of results from PCA
       * ts_res : dict of results from tsdiffana

    Examples
    --------
    >>> import nipy as ni
    >>> from nipy.testing import funcfile
    >>> img = ni.load_image(funcfile)
    >>> screen_res = screen(img)
    >>> screen_res['mean'].ndim
    3
    >>> screen_res['pca'].ndim
    4
    '''
    if img4d.ndim != 4:
        raise ValueError('Expecting a 4d image')
    data = np.asarray(img4d)
    cmap = img4d.coordmap
    cmap_3d = drop_io_dim(cmap, 't')
    screen_res = {}
    # standard processed images
    screen_res['mean'] = Image(np.mean(data, axis=-1), cmap_3d)
    screen_res['std'] = Image(np.std(data, axis=-1), cmap_3d)
    screen_res['max'] = Image(np.max(data, axis=-1), cmap_3d)
    screen_res['min'] = Image(np.min(data, axis=-1), cmap_3d)
    # PCA
    screen_res['pca_res'] = pca.pca(data,
                                    axis=-1,
                                    standardize=False,
                                    ncomp=ncomp)
    cmap_4d = append_io_dim(cmap_3d, 'l' , 'component')
    screen_res['pca'] = Image(screen_res['pca_res']['basis_projections'],
                              cmap_4d)
    # tsdiffana
    screen_res['ts_res'] = time_slice_diffs(data)
    return screen_res
