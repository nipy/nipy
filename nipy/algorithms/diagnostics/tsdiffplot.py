''' plot tsdiffana parameters '''

import numpy as np

import nipy
from .tsdiffana import time_slice_diffs


def plot_tsdiffs(results):
    ''' Plotting routine for time series difference metrics

    Requires matplotlib
    
    Parameters
    ----------
    results : dict
       Results of format returned from
       :func:`nipy.algorithms.diagnostics.time_slice_diff`
    
    '''
    T = len(results['volume_means'])
    S = results['slice_mean_diff2'].shape[1]
    mean_means = np.mean(results['volume_means'])
    scaled_slice_diff = results['slice_mean_diff2'] / mean_means

    import pylab
    from pylab import figure, plot, subplot
    n_plots = 4
    fig = figure()

    def xmax_labels(val, xlabel, ylabel):
        xlims = pylab.axis()
        pylab.axis((0, val) + xlims[2:])
        pylab.xlabel(xlabel)
        pylab.ylabel(ylabel)
        
    # plot of mean volume variance
    subplot(n_plots, 1, 1)
    plot(results['volume_mean_diff2'] / mean_means)
    xmax_labels(T-1, 'Difference image number', 'Scaled variance')
    # plot of diff by slice
    subplot(n_plots, 1, 2)
    plot(scaled_slice_diff, 'x')
    xmax_labels(T-1,
                'Difference image number',
                'Slice by slice variance')
    # mean intensity
    subplot(n_plots, 1, 3)
    plot(results['volume_means'] / mean_means)
    xmax_labels(T,
                'Image number',
                'Scaled mean voxel intensity')
    # slice plots min max mean
    subplot(n_plots, 1, 4)
    pylab.hold(True)
    plot(np.mean(scaled_slice_diff, 0), 'k')
    plot(np.min(scaled_slice_diff, 0), 'b')
    plot(np.max(scaled_slice_diff, 0), 'r')
    pylab.hold(False)
    xmax_labels(S+1,
                'Slice number',
                'Max/mean/min slice variation')
    
    # show the plot
    pylab.show()
    

def plot_tsdiffs_image(img):
    if isinstance(img, basestring):
        img = nipy.load_image(img)
    res = time_slice_diffs(img)
    plot_tsdiffs(res)

