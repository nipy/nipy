# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' plot tsdiffana parameters '''

import numpy as np

from nibabel import AnalyzeHeader

import nipy
from .timediff import time_slice_diffs, time_slice_diffs_image


def plot_tsdiffs(results, axes=None):
    ''' Plotting routine for time series difference metrics

    Requires matplotlib

    Parameters
    ----------
    results : dict
       Results of format returned from
       :func:`nipy.algorithms.diagnostics.time_slice_diff`
    '''
    import matplotlib.pyplot as plt
    T = len(results['volume_means'])
    S = results['slice_mean_diff2'].shape[1]
    mean_means = np.mean(results['volume_means'])
    scaled_slice_diff = results['slice_mean_diff2'] / mean_means

    if axes is None:
        n_plots = 4
        fig = plt.figure()
        fig.set_size_inches([10,10])
        axes = [plt.subplot(n_plots, 1, i+1) for i in range(n_plots)]

    def xmax_labels(ax, val, xlabel, ylabel):
        xlims = ax.axis()
        ax.axis((0, val) + xlims[2:])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # plot of mean volume variance
    ax = axes[0]
    ax.plot(results['volume_mean_diff2'] / mean_means)
    xmax_labels(ax, T-1, 'Difference image number', 'Scaled variance')

    # plot of diff by slice
    ax = axes[1]
    #Set up the color map for the different slices:
    X, Y = np.meshgrid(np.arange(scaled_slice_diff.shape[0]),
                       np.arange(scaled_slice_diff.shape[1]))

    # Use HSV in order to code the slices from bottom to top:
    ax.scatter(X.T.ravel(),scaled_slice_diff.ravel(),
               c=Y.T.ravel(),cmap=plt.cm.hsv,
               alpha=0.2)

    xmax_labels(ax, T-1,
                'Difference image number',
                'Slice by slice variance')

    # mean intensity
    ax = axes[2]
    ax.plot(results['volume_means'] / mean_means)
    xmax_labels(ax, T,
                'Image number',
                'Scaled mean \n voxel intensity')

    # slice plots min max mean
    ax = axes[3]
    ax.hold(True)
    ax.plot(np.mean(scaled_slice_diff, 0), 'k')
    ax.plot(np.min(scaled_slice_diff, 0), 'b')
    ax.plot(np.max(scaled_slice_diff, 0), 'r')
    ax.hold(False)
    xmax_labels(ax, S+1,
                'Slice number',
                'Max/mean/min \n slice variation')
    return axes


@np.deprecate_with_doc('Please see docstring for alternative code')
def plot_tsdiffs_image(img, axes=None, show=True):
    ''' Plot time series diagnostics for image

    This function is deprecated; please use something like::

        results = time_slice_diff_image(img, slice_axis=2)
        plot_tsdiffs(results)

    instead.

    Parameters
    ----------
    img : image-like or filename str
       image on which to do diagnostics
    axes : None or sequence, optional
       Axes on which to plot the diagnostics.  If None, then we create a figure
       and subplots for the plots.  Sequence should have length
       >=4.
    show : {True, False}, optional
       If True, show the figure after plotting it

    Returns
    -------
    axes : Matplotlib axes
       Axes on which we have done the plots.   Will be same as `axes` input if
       `axes` input was not None
    '''
    if isinstance(img, basestring):
        title = img
    else:
        title = 'Difference plots'
    img = nipy.as_image(img)
    res = time_slice_diffs(img)
    axes = plot_tsdiffs(res, axes)
    axes[0].set_title(title)
    if show:
        # show the plot
        import matplotlib.pyplot as plt
        plt.show()
    return axes


def tsdiffana_cmd(fname, time_axis=None, slice_axis=None):
    """ Generate tsdiffana plots for image file `fname`

    Parameters
    ----------
    fname : str
       filename of image on which to do diagnostics
    time_axis : None or str or int, optional
        Axis indexing time-points. None is default, will be replaced by a value
        of 't'. If `time_axis` is an integer, gives the index of the input
        (domain) axis of `img`. If `time_axis` is a str, can be an input
        (domain) name, or an output (range) name, that maps to an input
        (domain) name.
    slice_axis : None or str or int, optional
        Axis indexing MRI slices. If `slice_axis` is an integer, gives the
        index of the input (domain) axis of `img`. If `slice_axis` is a str,
        can be an input (domain) name, or an output (range) name, that maps to
        an input (domain) name.  If None (the default) then 1) try the name
        'slice' to select the axis - if this fails, and `fname` refers to an
        Analyze type image (such as Nifti), then 2) default to the third image
        axis, otherwise 3) raise a ValueError

    Returns
    -------
    axes : Matplotlib axes
       Axes on which we have done the plots.
    """
    # Check whether this is an Analyze-type image
    img = nipy.load_image(fname)
    # Check for axes
    if not time_axis is None:
        # Try converting to an integer in case that was what was passed
        try:
            time_axis = int(time_axis)
        except ValueError:
            # Maybe a string
            pass
    else: # was None
        time_axis = 't'
    if not slice_axis is None:
        # Try converting to an integer in case that was what was passed
        try:
            slice_axis = int(slice_axis)
        except ValueError:
            # Maybe a string
            pass
    else: # slice axis was None - search for default
        input_names = img.coordmap.function_domain.coord_names
        is_analyze = ('header' in img.metadata and
                      isinstance(img.metadata['header'], AnalyzeHeader))
        if 'slice' in input_names:
            slice_axis = 'slice'
        elif is_analyze and img.ndim == 4:
            slice_axis = 2
        else:
            raise ValueError('No slice axis specified, not analyze type '
                             'image; refusing to guess')
    results = time_slice_diffs_image(img, time_axis, slice_axis)
    return plot_tsdiffs(results)
