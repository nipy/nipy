"""
Example of script to parcellate the data from one subject,
using various algorithms.

Note that it can take some time.

author: Bertrand Thirion, 2005-2009
"""
print __doc__

import numpy as np
import os.path as op
import time
import tempfile

#from nipy.io.imageformats import load, save, Nifti1Image 
#import nipy.neurospin.graph as fg
#import nipy.neurospin.graph.field as ff
#from nipy.neurospin.clustering.hierarchical_clustering import \
#     ward_segment, ward_quick_segment
#import nipy.neurospin.spatial_models.parcellation as Pa
from nipy.neurospin.spatial_models.parcel_io import one_subj_parcellation

import get_data_light
get_data_light.getIt()

# ------------------------------------
# 1. Get the data (mask+functional image)
# take several experimental conditions
# time courses could be used instead

nbeta = [29]
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data'))
MaskImage = op.join(data_dir,'mask.nii.gz')
betas = [op.join(data_dir,'spmT_%04d.nii.gz'%n) for n in nbeta]

# set the parameters
nbparcel = 500
mu = 10
nn = 6
write_dir = tempfile.mkdtemp()
verbose = 1


lpa = one_subj_parcellation(MaskImage, betas, nbparcel, nn, 'gkm', 
                            write_dir, mu, verbose)
lpa = one_subj_parcellation(MaskImage, betas, nbparcel, nn, 'ward', 
                            write_dir, mu, verbose)
lpa = one_subj_parcellation(MaskImage, betas, nbparcel, nn, 'ward_and_gkm', 
                            write_dir, mu, verbose)
