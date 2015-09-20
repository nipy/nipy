""" Implementation of diagnostic command line tools

Tools are:

* nipy_diagnose
* nipy_tsdiffana

This module has the logic for each command.

The command script files deal with argument parsing and any custom imports.
The implementation here accepts the ``args`` object from ``argparse`` and does
the work.
"""
from __future__ import absolute_import
from os.path import split as psplit, join as pjoin

import numpy as np

from nibabel import AnalyzeHeader
from nibabel.filename_parser import splitext_addext

import nipy

from .tsdiffplot import plot_tsdiffs
from .timediff import time_slice_diffs_image
from .screens import screen, write_screen_res


def parse_fname_axes(img_fname, time_axis, slice_axis):
    """ Load `img_fname`, check `time_axis`, `slice_axis` or use default

    Parameters
    ----------
    img_fname : str
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
    img : ``Image`` instance
        Image as loaded from `img_fname`
    time_axis : int or str
        Time axis, possibly filled with default
    slice_axis : int or str
        Slice axis, possibly filled with default
    """
    # Check whether this is an Analyze-type image
    img = nipy.load_image(img_fname)
    # Check for axes
    if time_axis is not None:
        # Try converting to an integer in case that was what was passed
        try:
            time_axis = int(time_axis)
        except ValueError:
            # Maybe a string
            pass
    else: # was None
        time_axis = 't'
    if slice_axis is not None:
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
    return img, time_axis, slice_axis


def tsdiffana(args):
    """ Generate tsdiffana plots from command line params `args`

    Parameters
    ----------
    args : object
        object with attributes

        * filename : str - 4D image filename
        * out_file : str - graphics file to write to instead of leaving
          graphics on screen
        * time_axis : str - name or number of time axis in `filename`
        * slice_axis : str - name or number of slice axis in `filename`
        * write_results : bool - if True, write images and plots to files
        * out_path : None or str - path to which to write results
        * out_fname_label : None or filename - suffix of output results files

    Returns
    -------
    axes : Matplotlib axes
       Axes on which we have done the plots.
    """
    if args.out_file is not None and args.write_results:
        raise ValueError("Cannot have OUT_FILE and WRITE_RESULTS options "
                         "together")
    img, time_axis, slice_axis = parse_fname_axes(args.filename,
                                                  args.time_axis,
                                                  args.slice_axis)
    results = time_slice_diffs_image(img, time_axis, slice_axis)
    axes = plot_tsdiffs(results)
    if args.out_file is None and not args.write_results:
        # interactive mode
        return axes
    if args.out_file is not None:
        # plot only mode
        axes[0].figure.savefig(args.out_file)
        return axes
    # plot and images mode
    froot, ext, addext = splitext_addext(args.filename)
    fpath, fbase = psplit(froot)
    fpath = fpath if args.out_path is None else args.out_path
    fbase = fbase if args.out_fname_label is None else args.out_fname_label
    axes[0].figure.savefig(pjoin(fpath, 'tsdiff_' + fbase + '.png'))
    # Save image volumes
    for key, prefix in (('slice_diff2_max_vol', 'dv2_max_'),
                        ('diff2_mean_vol', 'dv2_mean_')):
        fname = pjoin(fpath, prefix + fbase + ext + addext)
        nipy.save_image(results[key], fname)
    # Save time courses into npz
    np.savez(pjoin(fpath, 'tsdiff_' + fbase + '.npz'),
             volume_means=results['volume_means'],
             slice_mean_diff2=results['slice_mean_diff2'],
            )
    return axes


def diagnose(args):
    """ Calculate, write results from diagnostic screen

    Parameters
    ----------
    args : object
        object with attributes:

        * filename : str - 4D image filename
        * time_axis : str - name or number of time axis in `filename`
        * slice_axis : str - name or number of slice axis in `filename`
        * out_path : None or str - path to which to write results
        * out_fname_label : None or filename - suffix of output results files
        * ncomponents : int - number of PCA components to write images for

    Returns
    -------
    res : dict
        Results of running :func:`screen` on `filename`
    """
    img, time_axis, slice_axis = parse_fname_axes(args.filename,
                                                  args.time_axis,
                                                  args.slice_axis)
    res = screen(img, args.ncomponents, time_axis, slice_axis)
    froot, ext, addext = splitext_addext(args.filename)
    fpath, fbase = psplit(froot)
    fpath = fpath if args.out_path is None else args.out_path
    fbase = fbase if args.out_fname_label is None else args.out_fname_label
    write_screen_res(res, fpath, fbase, ext + addext)
    return res
