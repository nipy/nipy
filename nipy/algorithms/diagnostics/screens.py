# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Diagnostic 4d image screen '''
from os.path import join as pjoin

import numpy as np

from ...core.api import Image, drop_io_dim, append_io_dim
from ...io.api import save_image
from ..utils import pca
from .timediff import time_slice_diffs
from .tsdiffplot import plot_tsdiffs


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
       * pca : 4D image of PCA component images
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
    cmap_4d = append_io_dim(cmap_3d, 'l' , 't')
    screen_res['pca'] = Image(screen_res['pca_res']['basis_projections'],
                              cmap_4d)
    # tsdiffana
    screen_res['ts_res'] = time_slice_diffs(data)
    return screen_res


def write_screen_res(res, out_path, out_root,
                     out_img_ext='.nii',
                     pcnt_var_thresh=0.1):
    ''' Write results from ``screen`` to disk as images
    
    Parameters
    ----------
    res : dict
       output from ``screen`` function
    out_path : str
       directory to which to write output images
    out_root : str
       part of filename between image-specific prefix and image-specific
       extension to use for writing images
    out_img_ext : str, optional
       extension (identifying image type) to which to write volume
       images.  Default is '.nii'
    pcnt_var_thresh : float, optional
       threshold below which we do not plot percent variance explained
       by components; default is 0.1.  This removes the long tail from
       percent variance plots.

    Returns
    -------
    None
    '''
    import matplotlib.pyplot as plt
    # save volume images
    for key in ('mean', 'min', 'max', 'std', 'pca'):
        fname = pjoin(out_path, '%s_%s%s' % (key,
                                             out_root,
                                             out_img_ext))
        save_image(res[key], fname)
    # plot, save component time courses
    ncomp = res['pca'].shape[-1]
    vectors = res['pca_res']['basis_vectors']
    plt.figure()
    for c in range(ncomp):
        plt.subplot(ncomp, 1, c+1)
        plt.plot(vectors[:,c])
        plt.axis('tight')
    plt.suptitle(out_root + ': PCA basis vectors')
    plt.savefig(pjoin(out_path, 'components_%s.png' % out_root))
    # plot percent variance
    plt.figure()
    pcnt_var = res['pca_res']['pcnt_var']
    plt.plot(pcnt_var[pcnt_var >= pcnt_var_thresh])
    plt.axis('tight')
    plt.suptitle(out_root + ': PCA percent variance')
    plt.savefig(pjoin(out_path, 'pcnt_var_%s.png' % out_root))    
    # plot tsdiffana
    plt.figure()
    axes = [plt.subplot(4, 1, i+1) for i in range(4)]
    plot_tsdiffs(res['ts_res'], axes)
    plt.suptitle(out_root + ': tsdiffana')
    plt.savefig(pjoin(out_path, 'tsdiff_%s.png' % out_root))
