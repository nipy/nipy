# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Support utilities for ds105 example, mostly path management.

The purpose of separating these is to keep the main example code as readable as
possible and focused on the experimental modeling and analysis, rather than on
local file management issues.

Requires matplotlib
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function # Python 2/3 compatibility

# Stdlib
import os
from os import makedirs, listdir
from os.path import exists, abspath, isdir, join as pjoin, splitext

# Third party
import numpy as np
from matplotlib.mlab import csv2rec

# From NIPY
from nipy.io.api import load_image

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

# We assume that there is a directory holding the data and it's local to this
# code.  Users can either keep a copy here or a symlink to the real location on
# disk of the data.
DATADIR = 'ds105_data'

# Sanity check
if not os.path.isdir(DATADIR):
    e="The data directory %s must exist and contain the ds105 data." % DATADIR
    raise IOError(e)

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

# Path management utilities
def load_image_ds105(*path):
    """Return a NIPY image from a set of path components.
    """
    return load_image(pjoin(DATADIR, *path))


def subj_des_con_dirs(design, contrast, subjects=range(1,7)):
    """Return a list of subject directories with this `design` and `contrast`

    Parameters
    ----------
    design : {'standard'} 
    contrast : str
    subjects : list, optional
        which subjects

    Returns
    -------
    con_dirs : list
        list of directories matching `design` and `contrast`
    """
    rootdir = DATADIR
    con_dirs = []
    for s in range(nsub):
        f = pjoin(rootdir, "sub%03d" % s, "model", design, "fixed", contrast)
        if isdir(f):
            con_dirs.append(f)
    return con_dirs


def path_info_run(subj, run, design='standard'):
    """Construct path information dict for current subject/run.

    Parameters
    ----------
    subj : int
        subject number (1..6 inclusive)
    run : int
        run number (1..12 inclusive).
    design : str, optional
        which design to use, defaults to 'standard'
    Returns
    -------
    path_dict : dict
        a dict with all the necessary path-related keys, including 'rootdir',
        and 'design', where 'design' can have values 'event' or 'block'
        depending on which type of run this was for subject no `subj` and run no
        `run`
    """
    path_dict = {'subj': subj, 'run': run, 'design':design}
    rootdir = pjoin(DATADIR, "sub%(subj)03d", "model", "%(design)s") % path_dict
    path_dict['rootdir'] = rootdir
    path_dict['fsldir'] = pjoin(DATADIR, "sub%(subj)03d", "model", "model001") % path_dict
    return path_dict


def path_info_design(subj, design):
    """Construct path information dict for subject and design.

    Parameters
    ----------
    subj : int
        subject number (1..6 inclusive)
    design : {'standard'}
        type of design

    Returns
    -------
    path_dict : dict
        having keys 'rootdir', 'subj', 'design'
    """
    path_dict = {'subj': subj, 'design': design}
    rootdir = pjoin(DATADIR, "sub%(subj)03d", "model", "%(design)s") % path_dict
    path_dict['rootdir'] = rootdir
    path_dict['fsldir'] = pjoin(DATADIR, "sub%(subj)03d", "model", "model001") % path_dict
    return path_dict


def results_table(path_dict):
    """ Return precalculated results images for subject info in `path_dict`

    Parameters
    ----------
    path_dict : dict
        containing key 'rootdir'

    Returns
    -------
    rtab : dict
        dict with keys given by run directories for this subject, values being a
        list with filenames of effect and sd images.
    """
    # Which runs correspond to this design type?
    rootdir = path_dict['rootdir']
    runs = filter(lambda f: isdir(pjoin(rootdir, f)),
                  ['results_run%03d' % i for i in range(1,13)] )

    # Find out which contrasts have t-statistics,
    # storing the filenames for reading below

    results = {}

    for rundir in runs:
        rundir = pjoin(rootdir, rundir)
        for condir in listdir(rundir):
            for stat in ['sd', 'effect']:
                fname_effect = abspath(pjoin(rundir, condir, 'effect.nii'))
                fname_sd = abspath(pjoin(rundir, condir, 'sd.nii'))
            if exists(fname_effect) and exists(fname_sd):
                results.setdefault(condir, []).append([fname_effect,
                                                       fname_sd])
    return results


def get_experiment(path_dict):
    """Get the record arrays for the experimental design.

    Parameters
    ----------
    path_dict : dict
        containing key 'rootdir', 'run', 'subj'

    Returns
    -------
    experiment, initial : Two record arrays.

    """
    # The following two lines read in the .csv files
    # and return recarrays, with fields
    # experiment: ['time', 'sentence', 'speaker']
    # initial: ['time', 'initial']

    rootdir = path_dict['rootdir']
    if not exists(pjoin(rootdir, "experiment_run%(run)03d.csv") % path_dict):
        e = "can't find design for subject=%(subj)d,run=%(subj)d" % path_dict
        raise IOError(e)

    experiment = csv2rec(pjoin(rootdir, "experiment_run%(run)03d.csv") % path_dict)

    return experiment


def get_fmri(path_dict):
    """Get the images for a given subject/run.

    Parameters
    ----------
    path_dict : dict
        containing key 'fsldir', 'run'

    Returns
    -------
    fmri : ndarray
    anat : NIPY image
    """
    fmri_im = load_image(
        pjoin("%(fsldir)s/task001_run%(run)03d.feat/filtered_func_data.nii.gz") % path_dict)
    return fmri_im


def ensure_dir(*path):
    """Ensure a directory exists, making it if necessary.

    Returns the full path."""
    dirpath = pjoin(*path)
    if not isdir(dirpath):
        makedirs(dirpath)
    return dirpath


def output_dir(path_dict, tcons, fcons):
    """Get (and make if necessary) directory to write output into.

    Parameters
    ----------
    path_dict : dict
        containing key 'rootdir', 'run'
    tcons : sequence of str
        t contrasts
    fcons : sequence of str
        F contrasts
    """
    rootdir = path_dict['rootdir']
    odir = pjoin(rootdir, "results_run%(run)03d" % path_dict)
    ensure_dir(odir)
    for n in tcons:
        ensure_dir(odir,n)
    for n in fcons:
        ensure_dir(odir,n)
    return odir

def compare_results(subj, run, other_root, mask_fname):
    """ Find and compare calculated results images from a previous run

    This scipt checks that another directory containing results of this same
    analysis are similar in the sense of numpy ``allclose`` within a brain mask.

    Parameters
    ----------
    subj : int
        subject number (1..6)
    run : int
        run number (1..12)
    other_root : str
        path to previous run estimation
    mask_fname:
        path to a mask image defining area in which to compare differences
    """
    # Get information for this subject and run
    path_dict = path_info_run(subj, run)
    # Get mask
    msk = load_image(mask_fname).get_data().copy().astype(bool)
    # Get results directories for this run
    rootdir = path_dict['rootdir']
    res_dir = pjoin(rootdir, 'results_run%03d' % run)
    if not isdir(res_dir):
        return
    for dirpath, dirnames, filenames in os.walk(res_dir):
        for fname in filenames:
            froot, ext = splitext(fname)
            if froot in ('effect', 'sd', 'F', 't'):
                this_fname = pjoin(dirpath, fname)
                other_fname = this_fname.replace(DATADIR, other_root)
                if not exists(other_fname):
                    print(this_fname, 'present but ', other_fname, 'missing')
                    continue
                this_arr = load_image(this_fname).get_data()
                other_arr = load_image(other_fname).get_data()
                ok = np.allclose(this_arr[msk], other_arr[msk])
                if not ok and froot in ('effect', 'sd', 't'): # Maybe a sign flip
                    ok = np.allclose(this_arr[msk], -other_arr[msk])
                if not ok:
                    print('Difference between', this_fname, other_fname)


def compare_all(other_root, mask_fname):
    """ Run results comparison for all subjects and runs """
    for subj in range(1,7):
        for run in range(1, 13):
            compare_results(subj, run, other_root, mask_fname)
