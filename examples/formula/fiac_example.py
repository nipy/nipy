import pylab
import numpy as np
from StringIO import StringIO
import os, csv

from matplotlib.mlab import csv2rec, rec2csv

from nipy.fixes.scipy.stats.models.regression import OLSModel, ARModel
from nipy.modalities.fmri.fmristat import hrf as delay
from nipy.modalities.fmri import formula, design, hrf
import nipy.testing as nitest
from nipy.io.api import load_image, save_image
from nipy.core import api

def test_sanity():
    from nipy.modalities.fmri.fmristat.tests import FIACdesigns

    """
    Single subject fitting of FIAC model
    """

    # Based on file
    # subj3_evt_fonc1.txt
    # subj3_bloc_fonc3.txt

    for subj, run, dtype in [(3,1,'event'),
                             (3,3,'block')]:
        nvol = 191
        TR = 2.5 
        Tstart = 1.25

        volume_times = np.arange(nvol)*TR + Tstart
        volume_times_rec = formula.make_recarray(volume_times, 't')

        path_dict = {'subj':subj, 'run':run}
        if os.path.exists("fiac_example_data/fiac_%(subj)02d/block/initial_%(run)02d.csv" % path_dict):
            path_dict['design'] = 'block'
        else:
            path_dict['design'] = 'event'

        experiment = csv2rec("fiac_example_data/fiac_%(subj)02d/%(design)s/experiment_%(run)02d.csv" % path_dict)
        initial = csv2rec("fiac_example_data/fiac_%(subj)02d/%(design)s/initial_%(run)02d.csv" % path_dict)

        X_exper, cons_exper = design.event_design(experiment, volume_times_rec, hrfs=delay.spectral)
        X_initial, _ = design.event_design(initial, volume_times_rec, hrfs=[hrf.glover]) 
        X, cons = design.stack_designs((X_exper, cons_exper),
                                       (X_initial, {}))

        Xf = np.loadtxt(StringIO(FIACdesigns.designs[dtype]))
        for i in range(X.shape[1]):
            yield nitest.assert_true, (matchcol(X[:,i], Xf.T)[1] > 0.999)
        
def matchcol(col, X):
    """
    Find the column in X with the highest correlation with col.

    Used to find matching columns in fMRIstat's design with
    the design created by Protocol. Not meant as a generic
    helper function.
    """
    c = np.array([np.corrcoef(col, X[i])[0,1] for i in range(X.shape[0])])
    c = np.nan_to_num(c)
    return np.argmax(c), c.max()

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

    if os.path.exists("%(root)s/fiac%(subj)d/subj%(subj)d_evt_fonc%(run)d.txt" % {'root':root, 'subj':subj, 'run':run}):
        designtype = 'evt'
    else:
        designtype = 'bloc'

    # Fix the format of the specification so it is
    # more in the form of a 2-way ANOVA

    eventdict = {1:'SSt_SSp', 2:'SSt_DSp', 3:'DSt_SSp', 4:'DSt_DSp'}
    s = StringIO()
    w = csv.writer(s)
    w.writerow(['time', 'sentence', 'speaker'])

    specfile = "%(root)s/fiac%(subj)d/subj%(subj)d_%(design)s_fonc%(run)d.txt" % {'root':root, 'subj':subj, 'run':run, 'design':designtype}
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

    fname = "fiac_example_data/fiac_%(subj)02d/%(design)s/experiment_%(run)02d.csv" % {'root':root, 'subj':subj, 'run':run, 'design':designtype}
    rec2csv(d, fname)
    experiment = csv2rec(fname)

    fname = "fiac_example_data/fiac_%(subj)02d/%(design)s/initial_%(run)02d.csv" % {'root':root, 'subj':subj, 'run':run, 'design':designtype}
    rec2csv(b, fname)
    initial = csv2rec(fname)

    return d, b

event = [(0,3),(0,4)] # Sequences with all the (subj, run) event designs 
block = [(0,1),(0,2)] # Sequences with all the (subj, run) block designs 

def fit(subj, run):
    """
    Single subject fitting of FIAC model
    """

    # Number of volumes in the fMRI data
    nvol = 191

    # The TR of the experiment
    TR = 2.5 

    # The time of the first volume
    # XXX Matthew: FSL slicetimes the data so that the first
    # time point is in the midpoint [0,TR]. What does SPM do?
    Tstart = 1.25

    # The array of times corresponding to each 
    # volume in the fMRI data

    volume_times = np.arange(nvol)*TR + Tstart

    # This recarray of times has one column named 't'
    # It is used in the function design.event_design
    # to create the design matrices.

    volume_times_rec = formula.make_recarray(volume_times, 't')

    # This few lines find the .csv files that 
    # describe the experiment

    path_dict = {'subj':subj, 'run':run}
    if os.path.exists("fiac_example_data/fiac_%(subj)02d/block/initial_%(run)02d.csv" % path_dict):
        path_dict['design'] = 'block'
    else:
        path_dict['design'] = 'event'

    # XXX -- fix the paths with this template
    path_tpl = "fiac_example_data/fiac_%(subj)02d/%(design)s"

    # The following two lines read in the .csv files
    # and return recarrays, with fields
    # experiment: ['time', 'sentence', 'speaker']
    # initial: ['time', 'initial']

    experiment = csv2rec("fiac_example_data/fiac_%(subj)02d/%(design)s/experiment_%(run)02d.csv" % path_dict)
    initial = csv2rec("fiac_example_data/fiac_%(subj)02d/%(design)s/initial_%(run)02d.csv" % path_dict)

    # Create design matrices for the "initial" and "experiment" factors,
    # saving the default contrasts. 

    # The function event_design will create
    # design matrices, which in the case of "experiment"
    # will have num_columns = (# levels of speaker) * (# levels of sentence) * len(delay.spectral) = 2 * 2 * 2 = 8
    # For "initial", there will be (# levels of initial) * len([hrf.glover]) = 1 * 1 = 1

    # Here, delay.spectral is a sequence of 2 symbolic HRFs that 
    # are described in 
    # 
    # Liao, C.H., Worsley, K.J., Poline, J-B., Aston, J.A.D., Duncan, G.H.,
    #    Evans, A.C. (2002). \'Estimating the delay of the response in fMRI
    #    data.\' NeuroImage, 16:593-606.

    # The contrasts, cons_exper,
    # is a dictionary with keys: ['constant_0', 'constant_1', 'speaker_0', speaker_1',
    # 'sentence_0', 'sentence_1', 'sentence:speaker_0', 'sentence:speaker_1']
    # representing the four default contrasts: constant, main effects + interactions,
    # each convolved with 2 HRFs in delay.spectral. Its values
    # are matrices with 8 columns.

    X_exper, cons_exper = design.event_design(experiment, volume_times_rec, hrfs=delay.spectral)

    # The contrasts for 'initial' are ignored 
    # as they are "uninteresting" and are included
    # in the model as confounds.

    X_initial, _ = design.event_design(initial, volume_times_rec, hrfs=[hrf.glover]) 

    # In addition to factors, there is typically a "drift" term
    # In this case, the drift is a natural cubic spline with
    # a not at the midpoint (volume_times.mean())

    vt = volume_times # shorthand
    drift = np.array([vt**i for i in range(4)] + [(vt-vt.mean())**3 * (np.greater(vt, vt.mean()))])
    for i in range(drift.shape[0]):
        drift[i] /= drift[i].max()

    # We transpose the drift so that its shape is (nvol,5) so that it will have
    # the same number of rows as X_initial and X_exper.
        
    drift = drift.T

    # Stack all the designs, keeping the new contrasts
    # which has the same keys as cons_exper, but its
    # values are arrays with 15 columns, with the non-zero
    # entries matching the columns of X corresponding to X_exper

    X, cons = design.stack_designs((X_exper, cons_exper),
                                   (X_initial, {}),
                                   (drift, {}))

    # The default contrasts are all t-statistics.
    # We may want to output F-statistics for
    # 'speaker', 'sentence', 'speaker:sentence' based
    # on the two coefficients, one for each HRF in delay.spectral

    cons['speaker'] = np.vstack([cons['speaker_0'], cons['speaker_1']])
    cons['sentence'] = np.vstack([cons['sentence_0'], cons['sentence_1']])
    cons['sentence:speaker'] = np.vstack([cons['sentence:speaker_0'], 
                                          cons['sentence:speaker_1']])

    # At this point, we're almost ready to fit a model
    # Load in the fMRI data, saving it as an array
    # It is transposed to have time as the first dimension,
    # i.e. fmri[t] gives the t-th volume.

    fmri = np.array(load_image("fiac_example_data/fiac_%(subj)02d/%(design)s/swafunctional_%(run)02d.nii" % path_dict))
    fmri = np.transpose(fmri, [3,2,1,0])
    anat = load_image("fiac_example_data/fiac_%(subj)02d/wanatomical.nii" % path_dict)
                   
    nvol, volshape = fmri.shape[0], fmri.shape[1:] 
    nslice, sliceshape = volshape[0], volshape[1:]

    # XXX Matthew: can you output a brain mask -- it would save
    # having to fit the model everywhere
    #     mask = np.array(load_image("fiac_example_data/fiac_%(subj)02d/mask.nii" % path_dict))
    #     mask_a = np.transpose(np.array(mask), [2,1,0])
    mask_a = 1 # XXX change this once we have a mask

    # The model is a two-stage model, the first stage being an OLS (ordinary least squares) fit,
    # whose residuals are used to estimate an AR(1) parameter for each voxel.

    m = OLSModel(X)
    ar1 = np.zeros(volshape)

    # Fit the model, storing an estimate of an AR(1) parameter at each voxel

    for s in range(nslice):
        d = np.array(fmri[:,s])
        flatd = d.reshape((d.shape[0], np.product(d.shape[1:])))
        result = m.fit(flatd)
        ar1[s] = ((result.resid[1:] * result.resid[:-1]).sum(0) / (result.resid**2).sum(0)).reshape(sliceshape)

    # We round ar1 to nearest one-hundredth
    # and group voxels by their rounded ar1 value,
    # fitting an AR(1) model to each batch of voxels.

    # XXX smooth here?
    # ar = smooth(ar, 8.0)

    ar1 *= 100
    ar1 = ar1.astype(np.int) / 100.

    # We split the contrasts into F-tests and t-tests.

    fcons = {}; tcons = {}
    for n, v in cons.items():
        v = np.squeeze(v)
        if v.ndim == 1:
            tcons[n] = v
        else:
            fcons[n] = v

    # Setup a dictionary to hold all the output

    output = {}
    for n in tcons:
        tempdict = {}
        for v in ['sd', 't', 'effect']:
            tempdict[v] = np.zeros(volshape)
        output[n] = tempdict
    
    for n in fcons:
        output[n] = np.zeros(volshape)

    # Loop over the unique values of ar1

    for val in np.unique(ar1):
        armask = np.equal(ar1, val)
        armask *= mask_a
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

            
    odir = "fiac_example_data/fiac_%(subj)02d/%(design)s/results_%(run)02d"  \
        % path_dict
    os.system('mkdir -p %s' % odir)

    for n in fcons:
        im = api.Image(output[n], anat.coordmap.copy())
        os.system('mkdir -p %s/%s' % (odir, n))
        save_image(im, "%s/%s/F.nii" % (odir, n))

    for n in tcons:
        os.system('mkdir -p %s/%s' % (odir, n))
        for v in ['t', 'sd', 'effect']:
            im = api.Image(output[n][v], anat.coordmap.copy())
            save_image(im, '%s/%s/%s.nii' % (odir, n, v))

