"""
Example of a script that uses the BSA (Bayesian Structural Analysis)
-- nipy.neurospin.spatial_models.bayesian_structural_analysis --
module

Please adapt the image paths to make it work on your own data

Author : Bertrand Thirion, 2008-2009
"""

#autoindent
import numpy as np
import scipy.stats as st
import os.path as op
import tempfile

from nipy.neurospin.spatial_models.bsa_nifti import make_bsa_nifti
import get_data_light


# Get the data
get_data_light.getIt()
nbsubj = 12
nbeta = 29
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                 'group_t_images'))
mask_images = [op.join(data_dir,'mask_subj%02d.nii'%n)
               for n in range(nbsubj)]

betas =[ op.join(data_dir,'spmT_%04d_subj_%02d.nii'%(nbeta,n))
                 for n in range(nbsubj)]

# set various parameters
subj_id = range(12)
theta = float(st.t.isf(0.01,100))
dmax = 5.
ths = 2 # or nbsubj/4
thq = 0.9
verbose = 1
smin = 5
swd = tempfile.mkdtemp()
method='simple'

# call the function
AF,BF = make_bsa_nifti(mask_images, betas, theta, dmax,
                       ths,thq,smin,swd,method,subj_id, nbeta)

# Write the result. OK, this is only a temporary solution
import pickle
picname = op.join(swd,"AF_%04d.pic" %nbeta)
pickle.dump(AF, open(picname, 'w'), 2)
picname = op.join(swd,"BF_%04d.pic" %nbeta)
pickle.dump(BF, open(picname, 'w'), 2)

print "Wrote all the results in directory %s"%swd
