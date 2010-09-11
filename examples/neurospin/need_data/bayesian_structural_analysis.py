# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of a script that uses the BSA (Bayesian Structural Analysis)
i.e. nipy.neurospin.spatial_models.bayesian_structural_analysis
module

Author : Bertrand Thirion, 2008-2010
"""
print __doc__

#autoindent
from scipy import stats
import os.path as op
import tempfile

from nipy.neurospin.spatial_models.bsa_io import make_bsa_image
import get_data_light

# Get the data
get_data_light.get_it()
nbsubj = 12
nbeta = 29
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                 'group_t_images'))
mask_images = [op.join(data_dir,'mask_subj%02d.nii'%n)
               for n in range(nbsubj)]

betas =[ op.join(data_dir,'spmT_%04d_subj_%02d.nii'%(nbeta,n))
                 for n in range(nbsubj)]

# set various parameters
subj_id = ['%04d' %i for i in range(12)]
theta = float(stats.t.isf(0.01, 100))
dmax = 4.
ths = 0 #2# or nbsubj/4
thq = 0.95
verbose = 1
smin = 5
swd = tempfile.mkdtemp()
method = 'simple'
print 'method used:', method

# call the function
AF, BF = make_bsa_image(mask_images, betas, theta, dmax, ths, thq, smin, swd,
                        method, subj_id, '%04d' % nbeta, reshuffle=False)

# Write the result. OK, this is only a temporary solution
import pickle
picname = op.join(swd,"AF_%04d.pic" %nbeta)
pickle.dump(AF, open(picname, 'w'), 2)
picname = op.join(swd,"BF_%04d.pic" %nbeta)
pickle.dump(BF, open(picname, 'w'), 2)

print "Wrote all the results in directory %s" % swd
