# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Script that perform the first-level analysis of the FIAC dataset.

Author: Alexis Roche, Bertrand Thirion, 2009--2012
"""

import os.path as op
import numpy as np
import pylab as pl
import tempfile
from nibabel import load, save, Nifti1Image

from nipy.modalities.fmri.glm import GeneralLinearModel, data_scaling
from nipy.utils import example_data
from nipy.labs.viz import plot_map, cm

# -----------------------------------------------------------
# --------- Get the data -----------------------------------
#-----------------------------------------------------------

fmri_files = [example_data.get_filename('fiac', 'fiac0', run)
              for run in ['run1.nii.gz', 'run2.nii.gz']]
design_files = [example_data.get_filename('fiac', 'fiac0', run)
                for run in ['run1_design.npz', 'run2_design.npz']]
mask_file = example_data.get_filename('fiac', 'fiac0', 'mask.nii.gz')
affine = load(mask_file).get_affine()

# Get design matrix as numpy array
print('Loading design matrices...')
X = [np.load(f)['X'] for f in design_files]

# Get multi-run fMRI data
print('Loading fmri data...')
Y = [load(f) for f in fmri_files]

# Get mask image
print('Loading mask...')
mask = load(mask_file)
mask_array = mask.get_data() > 0

# GLM fitting
print('Starting fit...')
results = []
for x, y in zip(X, Y):
    # normalize the data to report effects in percent of the baseline
    data = y.get_data()[mask_array].T
    data, mean = data_scaling(data)
    # fit the glm 
    model = GeneralLinearModel(x)
    model.fit(data, 'ar1')
    results.append(model)

# make a mean volume for display
wmean = mask_array.astype(np.int16)
wmean[mask_array] = mean


def make_fiac_contrasts():
    """Specify some constrasts for the FIAC experiment"""
    con = {}
    # the design matrices of both runs comprise 13 columns
    # the first 5 columns of the design matrices correpond to the following
    # conditions: ["SSt-SSp", "SSt-DSp", "DSt-SSp", "DSt-DSp", "FirstSt"]
    p = 13

    def length_p_vector(con, p):
        return np.hstack((con, np.zeros(p - len(con))))

    con["SStSSp_minus_DStDSp"] = length_p_vector([1, 0, 0, - 1], p)
    con["DStDSp_minus_SStSSp"] = length_p_vector([- 1, 0, 0, 1], p)
    con["DSt_minus_SSt"] = length_p_vector([- 1, - 1, 1, 1], p)
    con["DSp_minus_SSp"] = length_p_vector([- 1, 1, - 1, 1], p)
    con["DSt_minus_SSt_for_DSp"] = length_p_vector([0, - 1, 0, 1], p)
    con["DSp_minus_SSp_for_DSt"] = length_p_vector([0, 0, - 1, 1], p)
    con["Deactivation"] = length_p_vector([- 1, - 1, - 1, - 1, 4], p)
    con["Effects_of_interest"] = np.eye(p)[:5]
    return con


# compute fixed effects of the two runs and compute related images
contrasts = make_fiac_contrasts()
write_dir = tempfile.mkdtemp()
print 'Computing contrasts...'
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print '  Contrast % 2i out of %i: %s' % (
        index + 1, len(contrasts), contrast_id)
    contrast_path = op.join(write_dir, '%s_z_map.nii' % contrast_id)
    write_array = mask_array.astype(np.float)
    ffx_z_map = (results[0].contrast(contrast_val) +
                 results[1].contrast(contrast_val)).z_score()
    write_array[mask_array] = ffx_z_map
    contrast_image = Nifti1Image(write_array, affine)
    save(contrast_image, contrast_path)

    vmax = max(- write_array.min(), write_array.max())
    vmin = - vmax
    plot_map(write_array, affine,
             anat=wmean, anat_affine=affine,
             cmap=cm.cold_hot,
             vmin=vmin,
             vmax=vmax,
             figure=10,
             threshold=2.5,
             black_bg=True)
    pl.savefig(op.join(write_dir, '%s_z_map.png' % contrast_id))
    pl.clf()

print "All the  results were witten in %s" % write_dir
