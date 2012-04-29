# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Full step-by-step example of fitting a GLM to experimental data and
visualizing the results.

More specifically,
1. A sequence of fMRI volumes are loaded
2. A design matrix describing all the effects related to the data is computed
3. a mask of the useful brain volume is computed
4. A GLM is applied to the dataset (effect/covariance,
   then contrast estimation)

Note that this corresponds to a single session

Author : Bertrand Thirion, 2010
"""
print __doc__

import numpy as np
import os.path as op
import pylab
import tempfile
import scipy.stats as st

from nipy.labs import compute_mask_files
from nibabel import load, save, Nifti1Image

from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.experimental_paradigm import \
    load_protocol_from_csv_file
from nipy.labs.viz import plot_map, cm
from nipy.algorithms.statistics.models.regression import OLSModel, ARModel

import get_data_light

#######################################
# Data and analysis parameters
#######################################

# volume mask
get_data_light.get_first_level_dataset()
data_path = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                 's12069_swaloc1_corr.nii.gz'))
paradigm_file = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                      'localizer_paradigm.csv'))

# timing
n_scans = 128
tr = 2.4

# paradigm
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
conditions = ['damier_H', 'damier_V', 'clicDaudio', 'clicGaudio', 'clicDvideo',
              'clicGvideo', 'calculaudio', 'calculvideo', 'phrasevideo',
              'phraseaudio']

# confounds
hrf_model = 'canonical with derivative'
drift_model = "cosine"
hfcut = 128

# write directory
swd = tempfile.mkdtemp()
print 'Computation will be performed in temporary directory: %s' % swd

########################################
# Design matrix
########################################

print 'Loading design matrix...'

paradigm = load_protocol_from_csv_file(paradigm_file).values()[0]

design_matrix = make_dmtx(frametimes, paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, hfcut=hfcut)

ax = design_matrix.show()
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')

pylab.savefig(op.join(swd, 'design_matrix.png'))
# design_matrix.write_csv(...)

########################################
# Mask the data
########################################

print 'Computing a brain mask...'
mask_path = op.join(swd, 'mask.nii')
mask_array = compute_mask_files(data_path, mask_path, False, 0.4, 0.9)

#########################################
# Specify the contrasts
#########################################

# simplest ones
contrasts = {}
contrast_id = conditions
for i in range(len(conditions)):
    contrasts['%s' % conditions[i]] = np.eye(len(design_matrix.names))[2 * i]

# and more complex/ interesting ones
contrasts["audio"] = contrasts["clicDaudio"] + contrasts["clicGaudio"] +\
                     contrasts["calculaudio"] + contrasts["phraseaudio"]
contrasts["video"] = contrasts["clicDvideo"] + contrasts["clicGvideo"] + \
                     contrasts["calculvideo"] + contrasts["phrasevideo"]
contrasts["left"] = contrasts["clicGaudio"] + contrasts["clicGvideo"]
contrasts["right"] = contrasts["clicDaudio"] + contrasts["clicDvideo"]
contrasts["computation"] = contrasts["calculaudio"] + contrasts["calculvideo"]
contrasts["sentences"] = contrasts["phraseaudio"] + contrasts["phrasevideo"]
contrasts["H-V"] = contrasts["damier_H"] - contrasts["damier_V"]
contrasts["V-H"] = contrasts["damier_V"] - contrasts["damier_H"]
contrasts["left-right"] = contrasts["left"] - contrasts["right"]
contrasts["right-left"] = contrasts["right"] - contrasts["left"]
contrasts["audio-video"] = contrasts["audio"] - contrasts["video"]
contrasts["video-audio"] = contrasts["video"] - contrasts["audio"]
contrasts["computation-sentences"] = contrasts["computation"] -  \
                                     contrasts["sentences"]
contrasts["reading-visual"] = contrasts["sentences"] * 2 - \
                              contrasts["damier_H"] - contrasts["damier_V"]

output = {}
for contrast_id in contrasts.keys():
    tempdict = {}
    for v in ['sd', 't', 'effect']:
        tempdict[v] = np.zeros(mask_array.sum())
    output[contrast_id] = tempdict


########################################
# Perform a GLM analysis
########################################

print 'Fitting a GLM (this takes time)...'
fmri_image = load(data_path)
Y = fmri_image.get_data()[mask_array]
X = design_matrix.matrix

m = OLSModel(X)
# Fit the model, storing an estimate of an AR(1) parameter at each voxel
result = m.fit(Y.T)
ar1 = ((result.resid[1:] * result.resid[:-1]).sum(0) /
          (result.resid ** 2).sum(0))
ar1 *= 100
ar1 = ar1.astype(np.int) / 100.


for val in np.unique(ar1):
    armask = np.equal(ar1, val)
    m = ARModel(X, val)
    d = Y[armask]
    results = m.fit(d.T)

    # Output the results for each contrast
    for (contrast_id, contrast_val) in contrasts.items():
        resT = results.Tcontrast(contrast_val)
        output[contrast_id]['sd'][armask] = resT.sd
        output[contrast_id]['t'][armask] = resT.t
        output[contrast_id]['effect'][armask] = resT.effect

#########################################
# Estimate the contrasts
#########################################

print 'Computing contrasts...'
for index, contrast_id in enumerate(contrasts):
    print '  Contrast % 2i out of %i: %s' % (
        index + 1, len(contrasts), contrast_id)
    contrast_path = op.join(swd, '%s_z_map.nii' % contrast_id)
    write_array = mask_array.astype(np.float)
    z_values = st.norm.isf(st.t.sf(output[contrast_id]['t'], result.df_resid))
    write_array[mask_array] = z_values
    contrast_image = Nifti1Image(write_array, fmri_image.get_affine())
    save(contrast_image, contrast_path)
    affine = fmri_image.get_affine()

    vmax = max(- write_array.min(), write_array.max())
    plot_map(write_array, affine,
             cmap=cm.cold_hot,
             vmin=- vmax,
             vmax=vmax,
             anat=None,
             figure=10,
             threshold=2.5)
    pylab.savefig(op.join(swd, '%s_z_map.png' % contrast_id))
    pylab.clf()


#########################################
# End
#########################################

print "All the  results were witten in %s" % swd

plot_map(write_array, affine,
                cmap=cm.cold_hot,
                vmin=- vmax,
                vmax=vmax,
                anat=None,
                figure=10,
                threshold=3)

"""
plot_map(write_array, affine,
                cmap=cm.cold_hot,
                vmin=-vmax,
                vmax=vmax,
                anat=None,
                figure=10,
                threshold=3, do3d=True)

from nipy.labs import viz3d
viz3d.plot_map_3d(write_array, affine,
                cmap=cm.cold_hot,
                vmin=-vmax,
                vmax=vmax,
                anat=None,
                threshold=3)
"""
pylab.show()
