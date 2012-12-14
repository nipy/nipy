# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Example analyzing the FIAC dataset with NIPY.

* Single run models with per-voxel AR(1).
* Cross-run, within-subject models with optimal effect estimates.
* Cross-subject models using fixed / random effects variance ratios.
* Permutation testing for inference on cross-subject result.

See ``parallel_run.py`` for a rig to run these analysis in parallel using the
IPython parallel machinery.

This script needs the pre-processed FIAC data.  See ``README.txt`` and
``fiac_util.py`` for details.

See ``examples/labs/need_data/first_level_fiac.py`` for an alternative approach
to some of these analyses.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function # Python 2/3 compatibility

# Stdlib
from tempfile import NamedTemporaryFile
from os.path import join as pjoin
from copy import copy
import warnings

# Third party
import numpy as np

# From NIPY
from nipy.algorithms.statistics.api import (OLSModel, ARModel, make_recarray,
                                            isestimable)
from nipy.modalities.fmri.fmristat import hrf as delay
from nipy.modalities.fmri import design, hrf
from nipy.io.api import load_image, save_image
from nipy.core import api
from nipy.core.api import Image
from nipy.core.image.image import rollimg

from nipy.algorithms.statistics import onesample

# Local
import fiac_util as futil
reload(futil)  # while developing interactively

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

SUBJECTS = tuple(range(5) + range(6, 16)) # No data for subject 5
RUNS = tuple(range(1, 5))
DESIGNS = ('event', 'block')
CONTRASTS = ('speaker_0', 'speaker_1',
             'sentence_0', 'sentence_1',
             'sentence:speaker_0',
             'sentence:speaker_1')

GROUP_MASK = futil.load_image_fiac('group', 'mask.nii')
TINY_MASK = np.zeros(GROUP_MASK.shape, np.bool)
TINY_MASK[30:32,40:42,30:32] = 1

#-----------------------------------------------------------------------------
# Public functions
#-----------------------------------------------------------------------------

# For group analysis

def run_model(subj, run):
    """
    Single subject fitting of FIAC model
    """
    #----------------------------------------------------------------------
    # Set initial parameters of the FIAC dataset
    #----------------------------------------------------------------------
    # Number of volumes in the fMRI data
    nvol = 191
    # The TR of the experiment
    TR = 2.5
    # The time of the first volume
    Tstart = 0.0
    # The array of times corresponding to each volume in the fMRI data
    volume_times = np.arange(nvol) * TR + Tstart
    # This recarray of times has one column named 't'.  It is used in the
    # function design.event_design to create the design matrices.
    volume_times_rec = make_recarray(volume_times, 't')
    # Get a path description dictionary that contains all the path data relevant
    # to this subject/run
    path_info = futil.path_info_run(subj,run)

    #----------------------------------------------------------------------
    # Experimental design
    #----------------------------------------------------------------------

    # Load the experimental description from disk.  We have utilities in futil
    # that reformat the original FIAC-supplied format into something where the
    # factorial structure of the design is more explicit.  This has already
    # been run once, and get_experiment_initial() will simply load the
    # newly-formatted design description files (.csv) into record arrays.
    experiment, initial = futil.get_experiment_initial(path_info)

    # Create design matrices for the "initial" and "experiment" factors, saving
    # the default contrasts.

    # The function event_design will create design matrices, which in the case
    # of "experiment" will have num_columns = (# levels of speaker) * (# levels
    # of sentence) * len(delay.spectral) = 2 * 2 * 2 = 8. For "initial", there
    # will be (# levels of initial) * len([hrf.glover]) = 1 * 1 = 1.

    # Here, delay.spectral is a sequence of 2 symbolic HRFs that are described
    # in:
    #
    # Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H.,
    #    Evans, A.C. (2002). \'Estimating the delay of the response in fMRI
    #    data.\' NeuroImage, 16:593-606.

    # The contrast definitions in ``cons_exper`` are a dictionary with keys
    # ['constant_0', 'constant_1', 'speaker_0', 'speaker_1', 'sentence_0',
    # 'sentence_1', 'sentence:speaker_0', 'sentence:speaker_1'] representing the
    # four default contrasts: constant, main effects + interactions, each
    # convolved with 2 HRFs in delay.spectral. For example, sentence:speaker_0
    # is the interaction of sentence and speaker convolved with the first (=0)
    # of the two HRF basis functions, and sentence:speaker_1 is the interaction
    # convolved with the second (=1) of the basis functions.

    # XXX use the hrf __repr__ for naming contrasts
    X_exper, cons_exper = design.event_design(experiment, volume_times_rec,
                                              hrfs=delay.spectral)

    # The contrasts for 'initial' are ignored as they are "uninteresting" and
    # are included in the model as confounds.
    X_initial, _ = design.event_design(initial, volume_times_rec,
                                       hrfs=[hrf.glover])

    # In addition to factors, there is typically a "drift" term. In this case,
    # the drift is a natural cubic spline with a not at the midpoint
    # (volume_times.mean())
    vt = volume_times # shorthand
    drift = np.array( [vt**i for i in range(4)] +
                      [(vt-vt.mean())**3 * (np.greater(vt, vt.mean()))] )
    for i in range(drift.shape[0]):
        drift[i] /= drift[i].max()

    # We transpose the drift so that its shape is (nvol,5) so that it will have
    # the same number of rows as X_initial and X_exper.
    drift = drift.T

    # There are helper functions to create these drifts: design.fourier_basis,
    # design.natural_spline.  Therefore, the above is equivalent (except for
    # the normalization by max for numerical stability) to
    #
    # >>> drift = design.natural_spline(t, [volume_times.mean()])

    # Stack all the designs, keeping the new contrasts which has the same keys
    # as cons_exper, but its values are arrays with 15 columns, with the
    # non-zero entries matching the columns of X corresponding to X_exper
    X, cons = design.stack_designs((X_exper, cons_exper),
                                   (X_initial, {}),
                                   (drift, {}))

    # Sanity check: delete any non-estimable contrasts
    for k in cons.keys():
        if not isestimable(cons[k], X):
            del(cons[k])
            warnings.warn("contrast %s not estimable for this run" % k)

    # The default contrasts are all t-statistics.  We may want to output
    # F-statistics for 'speaker', 'sentence', 'speaker:sentence' based on the
    # two coefficients, one for each HRF in delay.spectral

    cons['speaker'] = np.vstack([cons['speaker_0'], cons['speaker_1']])
    cons['sentence'] = np.vstack([cons['sentence_0'], cons['sentence_1']])
    cons['sentence:speaker'] = np.vstack([cons['sentence:speaker_0'],
                                          cons['sentence:speaker_1']])

    #----------------------------------------------------------------------
    # Data loading
    #----------------------------------------------------------------------

    # Load in the fMRI data, saving it as an array.  It is transposed to have
    # time as the first dimension, i.e. fmri[t] gives the t-th volume.
    fmri_im = futil.get_fmri(path_info) # an Image
    fmri_im = rollimg(fmri_im, 't', fix0=True)
    fmri = fmri_im.get_data() # now, it's an ndarray

    nvol, volshape = fmri.shape[0], fmri.shape[1:]
    nx, sliceshape = volshape[0], volshape[1:]

    #----------------------------------------------------------------------
    # Model fit
    #----------------------------------------------------------------------

    # The model is a two-stage model, the first stage being an OLS (ordinary
    # least squares) fit, whose residuals are used to estimate an AR(1)
    # parameter for each voxel.
    m = OLSModel(X)
    ar1 = np.zeros(volshape)

    # Fit the model, storing an estimate of an AR(1) parameter at each voxel
    for s in range(nx):
        d = np.array(fmri[:,s])
        flatd = d.reshape((d.shape[0], -1))
        result = m.fit(flatd)
        ar1[s] = ((result.resid[1:] * result.resid[:-1]).sum(0) /
                  (result.resid**2).sum(0)).reshape(sliceshape)

    # We round ar1 to nearest one-hundredth and group voxels by their rounded
    # ar1 value, fitting an AR(1) model to each batch of voxels.

    # XXX smooth here?
    # ar1 = smooth(ar1, 8.0)
    ar1 *= 100
    ar1 = ar1.astype(np.int) / 100.

    # We split the contrasts into F-tests and t-tests.
    # XXX helper function should do this
    fcons = {}; tcons = {}
    for n, v in cons.items():
        v = np.squeeze(v)
        if v.ndim == 1:
            tcons[n] = v
        else:
            fcons[n] = v

    # Setup a dictionary to hold all the output
    # XXX ideally these would be memmap'ed Image instances
    output = {}
    for n in tcons:
        tempdict = {}
        for v in ['sd', 't', 'effect']:
            tempdict[v] = np.memmap(NamedTemporaryFile(prefix='%s%s.nii'
                                    % (n,v)), dtype=np.float,
                                    shape=volshape, mode='w+')
        output[n] = tempdict

    for n in fcons:
        output[n] = np.memmap(NamedTemporaryFile(prefix='%s%s.nii'
                                    % (n,v)), dtype=np.float,
                                    shape=volshape, mode='w+')

    # Loop over the unique values of ar1
    for val in np.unique(ar1):
        armask = np.equal(ar1, val)
        m = ARModel(X, val)
        d = fmri[:,armask]
        results = m.fit(d)

        # Output the results for each contrast
        for n in tcons:
            resT = results.Tcontrast(tcons[n])
            output[n]['sd'][armask] = resT.sd
            output[n]['t'][armask] = resT.t
            output[n]['effect'][armask] = resT.effect
        for n in fcons:
            output[n][armask] = results.Fcontrast(fcons[n]).F

    # Dump output to disk
    odir = futil.output_dir(path_info,tcons,fcons)
    # The coordmap for a single volume in the time series
    vol0_map = fmri_im[0].coordmap
    for n in tcons:
        for v in ['t', 'sd', 'effect']:
            im = Image(output[n][v], vol0_map)
            save_image(im, pjoin(odir, n, '%s.nii' % v))
    for n in fcons:
        im = Image(output[n], vol0_map)
        save_image(im, pjoin(odir, n, "F.nii"))


def fixed_effects(subj, design):
    """ Fixed effects (within subject) for FIAC model

    Finds run by run estimated model results, creates fixed effects results
    image per subject.

    Parameters
    ----------
    subj : int
        subject number 0..15 inclusive
    design : {'block', 'event'}
        design type
    """
    # First, find all the effect and standard deviation images
    # for the subject and this design type
    path_dict = futil.path_info_design(subj, design)
    rootdir = path_dict['rootdir']
    # The output directory
    fixdir = pjoin(rootdir, "fixed")
    # Fetch results images from run estimations
    results = futil.results_table(path_dict)
    # Get our hands on the relevant coordmap to save our results
    coordmap = futil.load_image_fiac("fiac_%02d" % subj,
                                     "wanatomical.nii").coordmap
    # Compute the "fixed" effects for each type of contrast
    for con in results:
        fixed_effect = 0
        fixed_var = 0
        for effect, sd in results[con]:
            effect = load_image(effect).get_data()
            sd = load_image(sd).get_data()
            var = sd ** 2

            # The optimal, in terms of minimum variance, combination of the
            # effects has weights 1 / var
            #
            # XXX regions with 0 variance are set to 0
            # XXX do we want this or np.nan?
            ivar = np.nan_to_num(1. / var)
            fixed_effect += effect * ivar
            fixed_var += ivar

        # Now, compute the fixed effects variance and t statistic
        fixed_sd = np.sqrt(fixed_var)
        isd = np.nan_to_num(1. / fixed_sd)
        fixed_t = fixed_effect * isd

        # Save the results
        odir = futil.ensure_dir(fixdir, con)
        for a, n in zip([fixed_effect, fixed_sd, fixed_t],
                        ['effect', 'sd', 't']):
            im = api.Image(a, copy(coordmap))
            save_image(im, pjoin(odir, '%s.nii' % n))


def group_analysis(design, contrast):
    """ Compute group analysis effect, t, sd for `design` and `contrast`

    Saves to disk in 'group' analysis directory

    Parameters
    ----------
    design : {'block', 'event'}
    contrast : str
        contrast name
    """
    array = np.array # shorthand
    # Directory where output will be written
    odir = futil.ensure_dir(futil.DATADIR, 'group', design, contrast)

    # Which subjects have this (contrast, design) pair?
    subj_con_dirs = futil.subj_des_con_dirs(design, contrast)
    if len(subj_con_dirs) == 0:
        raise ValueError('No subjects for %s, %s' % (design, contrast))

    # Assemble effects and sds into 4D arrays
    sds = []
    Ys = []
    for s in subj_con_dirs:
        sd_img = load_image(pjoin(s, "sd.nii"))
        effect_img = load_image(pjoin(s, "effect.nii"))
        sds.append(sd_img.get_data())
        Ys.append(effect_img.get_data())
    sd = array(sds)
    Y = array(Ys)

    # This function estimates the ratio of the fixed effects variance
    # (sum(1/sd**2, 0)) to the estimated random effects variance
    # (sum(1/(sd+rvar)**2, 0)) where rvar is the random effects variance.

    # The EM algorithm used is described in:
    #
    # Worsley, K.J., Liao, C., Aston, J., Petre, V., Duncan, G.H.,
    #    Morales, F., Evans, A.C. (2002). \'A general statistical
    #    analysis for fMRI data\'. NeuroImage, 15:1-15
    varest = onesample.estimate_varatio(Y, sd)
    random_var = varest['random']

    # XXX - if we have a smoother, use
    # random_var = varest['fixed'] * smooth(varest['ratio'])

    # Having estimated the random effects variance (and possibly smoothed it),
    # the corresponding estimate of the effect and its variance is computed and
    # saved.

    # This is the coordmap we will use
    coordmap = futil.load_image_fiac("fiac_00","wanatomical.nii").coordmap

    adjusted_var = sd**2 + random_var
    adjusted_sd = np.sqrt(adjusted_var)

    results = onesample.estimate_mean(Y, adjusted_sd) 
    for n in ['effect', 'sd', 't']:
        im = api.Image(results[n], copy(coordmap))
        save_image(im, pjoin(odir, "%s.nii" % n))


def group_analysis_signs(design, contrast, mask, signs=None):
    """ Refit the EM model with a vector of signs.

    Used in the permutation tests.

    Returns the maximum of the T-statistic within mask

    Parameters
    ----------
    design: one of 'block', 'event'
    contrast: str
        name of contrast to estimate
    mask : ``Image`` instance or array-like
        image containing mask, or array-like
    signs: ndarray, optional
         Defaults to np.ones. Should have shape (*,nsubj)
         where nsubj is the number of effects combined in the group analysis.

    Returns
    -------
    minT: np.ndarray, minima of T statistic within mask, one for each
         vector of signs
    maxT: np.ndarray, maxima of T statistic within mask, one for each
         vector of signs
    """
    if api.is_image(mask):
        maska = mask.get_data()
    else:
        maska = np.asarray(mask)
    maska = maska.astype(np.bool)

    # Which subjects have this (contrast, design) pair?
    subj_con_dirs = futil.subj_des_con_dirs(design, contrast)

    # Assemble effects and sds into 4D arrays
    sds = []
    Ys = []
    for s in subj_con_dirs:
        sd_img = load_image(pjoin(s, "sd.nii"))
        effect_img = load_image(pjoin(s, "effect.nii"))
        sds.append(sd_img.get_data()[maska])
        Ys.append(effect_img.get_data()[maska])
    sd = np.array(sds)
    Y = np.array(Ys)

    if signs is None:
        signs = np.ones((1, Y.shape[0]))

    maxT = np.empty(signs.shape[0])
    minT = np.empty(signs.shape[0])

    for i, sign in enumerate(signs):
        signY = sign[:,np.newaxis] * Y
        varest = onesample.estimate_varatio(signY, sd)
        random_var = varest['random']

        adjusted_var = sd**2 + random_var
        adjusted_sd = np.sqrt(adjusted_var)

        results = onesample.estimate_mean(Y, adjusted_sd) 
        T = results['t']
        minT[i], maxT[i] = np.nanmin(T), np.nanmax(T)
    return minT, maxT


def permutation_test(design, contrast, mask=GROUP_MASK, nsample=1000):
    """
    Perform a permutation (sign) test for a given design type and
    contrast. It is a Monte Carlo test because we only sample nsample
    possible sign arrays.

    Parameters
    ----------
    design: str
        one of ['block', 'event']
    contrast : str
        name of contrast to estimate
    mask : ``Image`` instance or array-like, optional
        image containing mask, or array-like
    nsample: int, optional
        number of permutations

    Returns
    -------
    min_vals: np.ndarray
    max_vals: np.ndarray
    """
    subj_con_dirs = futil.subj_des_con_dirs(design, contrast)
    nsubj = len(subj_con_dirs)
    if nsubj == 0:
        raise ValueError('No subjects have %s, %s' % (design, contrast))
    signs = 2*np.greater(np.random.sample(size=(nsample, nsubj)), 0.5) - 1
    min_vals, max_vals = group_analysis_signs(design, contrast, mask, signs)
    return min_vals, max_vals


def run_run_models(subject_nos=SUBJECTS, run_nos = RUNS):
    """ Simple serial run of all the within-run models """
    for subj in subject_nos:
        for run in run_nos:
            try:
                run_model(subj, run)
            except IOError:
                print('Skipping subject %d, run %d' % (subj, run))


def run_fixed_models(subject_nos=SUBJECTS, designs=DESIGNS):
    """ Simple serial run of all the within-subject models """
    for subj in subject_nos:
        for design in designs:
            try:
                fixed_effects(subj, design)
            except IOError:
                print('Skipping subject %d, design %s' % (subj, design))


def run_group_models(designs=DESIGNS, contrasts=CONTRASTS):
    """ Simple serial run of all the across-subject models """
    for design in designs:
        for contrast in contrasts:
            group_analysis(design, contrast)


if __name__ == '__main__':
    pass
    # Sanity check while debugging
    #permutation_test('block','sentence_0',mask=TINY_MASK,nsample=3)
