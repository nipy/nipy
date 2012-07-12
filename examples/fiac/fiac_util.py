
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Support utilities for FIAC example, mostly path management.

The purpose of separating these is to keep the main example code as readable as
possible and focused on the experimental modeling and analysis, rather than on
local file management issues.

Requires matplotlib
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
# Stdlib
import os
from os import makedirs, listdir
from os.path import exists, abspath, isdir, join as pjoin, splitext
import csv
from StringIO import StringIO

# Third party
import numpy as np
from matplotlib.mlab import csv2rec, rec2csv

# From NIPY
from nipy.io.api import load_image

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

# We assume that there is a directory holding the data and it's local to this
# code.  Users can either keep a copy here or a symlink to the real location on
# disk of the data.
DATADIR = 'fiac_data'

# Sanity check
if not os.path.isdir(DATADIR):
    e="The data directory %s must exist and contain the FIAC data." % DATADIR
    raise IOError(e)

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

# Path management utilities
def load_image_fiac(*path):
    """Return a NIPY image from a set of path components.
    """
    return load_image(pjoin(DATADIR, *path))


def subj_des_con_dirs(design, contrast, nsub=16):
    """Return a list of subject directories with this `design` and `contrast`

    Parameters
    ----------
    design : {'event', 'block'}
    contrast : str
    nsub : int, optional
        total number of subjects

    Returns
    -------
    con_dirs : list
        list of directories matching `design` and `contrast`
    """
    rootdir = DATADIR
    con_dirs = []
    for s in range(nsub):
        f = pjoin(rootdir, "fiac_%02d" % s, design, "fixed", contrast)
        if isdir(f):
            con_dirs.append(f)
    return con_dirs


def path_info_run(subj, run):
    """Construct path information dict for current subject/run.

    Parameters
    ----------
    subj : int
        subject number (0..15 inclusive)
    run : int
        run number (1..4 inclusive).

    Returns
    -------
    path_dict : dict
        a dict with all the necessary path-related keys, including 'rootdir',
        and 'design', where 'design' can have values 'event' or 'block'
        depending on which type of run this was for subject no `subj` and run no
        `run`
    """
    path_dict = {'subj': subj, 'run': run}
    if exists(pjoin(DATADIR, "fiac_%(subj)02d",
                    "block", "initial_%(run)02d.csv") % path_dict):
        path_dict['design'] = 'block'
    else:
        path_dict['design'] = 'event'
    rootdir = pjoin(DATADIR, "fiac_%(subj)02d", "%(design)s") % path_dict
    path_dict['rootdir'] = rootdir
    return path_dict


def path_info_design(subj, design):
    """Construct path information dict for subject and design.

    Parameters
    ----------
    subj : int
        subject number (0..15 inclusive)
    design : {'event', 'block'}
        type of design

    Returns
    -------
    path_dict : dict
        having keys 'rootdir', 'subj', 'design'
    """
    path_dict = {'subj': subj, 'design': design}
    rootdir = pjoin(DATADIR, "fiac_%(subj)02d", "%(design)s") % path_dict
    path_dict['rootdir'] = rootdir
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
                  ['results_%02d' % i for i in range(1,5)] )

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


def get_experiment_initial(path_dict):
    """Get the record arrays for the experimental/initial designs.

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
    if not exists(pjoin(rootdir, "experiment_%(run)02d.csv") % path_dict):
        e = "can't find design for subject=%(subj)d,run=%(subj)d" % path_dict
        raise IOError(e)

    experiment = csv2rec(pjoin(rootdir, "experiment_%(run)02d.csv") % path_dict)
    initial = csv2rec(pjoin(rootdir, "initial_%(run)02d.csv") % path_dict)

    return experiment, initial


def get_fmri(path_dict):
    """Get the images for a given subject/run.

    Parameters
    ----------
    path_dict : dict
        containing key 'rootdir', 'run'

    Returns
    -------
    fmri : ndarray
    anat : NIPY image
    """
    fmri_im = load_image(
        pjoin("%(rootdir)s/swafunctional_%(run)02d.nii") % path_dict)
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
    odir = pjoin(rootdir, "results_%(run)02d" % path_dict)
    ensure_dir(odir)
    for n in tcons:
        ensure_dir(odir,n)
    for n in fcons:
        ensure_dir(odir,n)
    return odir


def test_sanity():
    from nipy.modalities.fmri import design, hrf
    import nipy.modalities.fmri.fmristat.hrf as fshrf
    from nipy.modalities.fmri.fmristat.tests import FIACdesigns
    from nipy.modalities.fmri.fmristat.tests.test_FIAC import matchcol
    from nipy.algorithms.statistics import formula
    from nose.tools import assert_true

    """
    Single subject fitting of FIAC model
    """

    # Based on file
    # subj3_evt_fonc1.txt
    # subj3_bloc_fonc3.txt

    for subj, run, design_type in [(3, 1, 'event'), (3, 3, 'block')]:
        nvol = 191
        TR = 2.5
        Tstart = 1.25

        volume_times = np.arange(nvol)*TR + Tstart
        volume_times_rec = formula.make_recarray(volume_times, 't')

        path_dict = {'subj':subj, 'run':run}
        if exists(pjoin(DATADIR, "fiac_%(subj)02d",
                        "block", "initial_%(run)02d.csv") % path_dict):
            path_dict['design'] = 'block'
        else:
            path_dict['design'] = 'event'

        experiment = csv2rec(pjoin(DATADIR, "fiac_%(subj)02d", "%(design)s", "experiment_%(run)02d.csv")
                             % path_dict)
        initial = csv2rec(pjoin(DATADIR, "fiac_%(subj)02d", "%(design)s", "initial_%(run)02d.csv")
                                % path_dict)

        X_exper, cons_exper = design.event_design(experiment,
                                                  volume_times_rec,
                                                  hrfs=fshrf.spectral)
        X_initial, _ = design.event_design(initial,
                                           volume_times_rec,
                                           hrfs=[hrf.glover])
        X, cons = design.stack_designs((X_exper, cons_exper), (X_initial, {}))

        # Get original fmristat design
        Xf = FIACdesigns.fmristat[design_type]
        # Check our new design can be closely matched to the original
        for i in range(X.shape[1]):
            # Columns can be very well correlated negatively or positively
            assert_true(abs(matchcol(X[:,i], Xf)[1]) > 0.999)


def rewrite_spec(subj, run, root = "/home/jtaylo/FIAC-HBM2009"):
    """
    Take a FIAC specification file and get two specifications
    (experiment, begin).

    This creates two new .csv files, one for the experimental
    conditions, the other for the "initial" confounding trials that
    are to be modelled out. 

    For the block design, the "initial" trials are the first
    trials of each block. For the event designs, the 
    "initial" trials are made up of just the first trial.

    """

    if exists(pjoin("%(root)s", "fiac%(subj)d", "subj%(subj)d_evt_fonc%(run)d.txt") % {'root':root, 'subj':subj, 'run':run}):
        designtype = 'evt'
    else:
        designtype = 'bloc'

    # Fix the format of the specification so it is
    # more in the form of a 2-way ANOVA

    eventdict = {1:'SSt_SSp', 2:'SSt_DSp', 3:'DSt_SSp', 4:'DSt_DSp'}
    s = StringIO()
    w = csv.writer(s)
    w.writerow(['time', 'sentence', 'speaker'])

    specfile = pjoin("%(root)s", "fiac%(subj)d", "subj%(subj)d_%(design)s_fonc%(run)d.txt") % {'root':root, 'subj':subj, 'run':run, 'design':designtype}
    d = np.loadtxt(specfile)
    for row in d:
        w.writerow([row[0]] + eventdict[row[1]].split('_'))
    s.seek(0)
    d = csv2rec(s)

    # Now, take care of the 'begin' event
    # This is due to the FIAC design

    if designtype == 'evt':
        b = np.array([(d[0]['time'], 1)], np.dtype([('time', np.float),
                                                    ('initial', np.int)]))
        d = d[1:]
    else:
        k = np.equal(np.arange(d.shape[0]) % 6, 0)
        b = np.array([(tt, 1) for tt in d[k]['time']], np.dtype([('time', np.float),
                                                                 ('initial', np.int)]))
        d = d[~k]

    designtype = {'bloc':'block', 'evt':'event'}[designtype]

    fname = pjoin(DATADIR, "fiac_%(subj)02d", "%(design)s", "experiment_%(run)02d.csv") % {'root':root, 'subj':subj, 'run':run, 'design':designtype}
    rec2csv(d, fname)
    experiment = csv2rec(fname)

    fname = pjoin(DATADIR, "fiac_%(subj)02d", "%(design)s", "initial_%(run)02d.csv") % {'root':root, 'subj':subj, 'run':run, 'design':designtype}
    rec2csv(b, fname)
    initial = csv2rec(fname)

    return d, b


def compare_results(subj, run, other_root, mask_fname):
    """ Find and compare calculated results images from a previous run

    This scipt checks that another directory containing results of this same
    analysis are similar in the sense of numpy ``allclose`` within a brain mask.

    Parameters
    ----------
    subj : int
        subject number (0..4, 6..15)
    run : int
        run number (1..4)
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
    res_dir = pjoin(rootdir, 'results_%02d' % run)
    if not isdir(res_dir):
        return
    for dirpath, dirnames, filenames in os.walk(res_dir):
        for fname in filenames:
            froot, ext = splitext(fname)
            if froot in ('effect', 'sd', 'F', 't'):
                this_fname = pjoin(dirpath, fname)
                other_fname = this_fname.replace(DATADIR, other_root)
                if not exists(other_fname):
                    print this_fname, 'present but ', other_fname, 'missing'
                    continue
                this_arr = load_image(this_fname).get_data()
                other_arr = load_image(other_fname).get_data()
                ok = np.allclose(this_arr[msk], other_arr[msk])
                if not ok and froot in ('effect', 'sd', 't'): # Maybe a sign flip
                    ok = np.allclose(this_arr[msk], -other_arr[msk])
                if not ok:
                    print 'Difference between', this_fname, other_fname


def compare_all(other_root, mask_fname):
    """ Run results comparison for all subjects and runs """
    for subj in range(5) + range(6, 16):
        for run in range(1, 5):
            compare_results(subj, run, other_root, mask_fname)
