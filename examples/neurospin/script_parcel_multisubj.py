"""
Example of script to parcellate mutli-subject data
author: Bertrand Thirion, 2005-2009
"""
import os.path as op
import cPickle
import tempfile

from nipy.neurospin.spatial_models.parcellation import Parcellation
from nipy.neurospin.spatial_models.parcel_io_nii import *
from nipy.neurospin.spatial_models.hierarchical_parcellation import hparcel
import get_data_light

# Get the data
get_data_light.getIt()
nbsubj = 12
subj_id = range(nbsubj)
numbeta = [29]
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                 'group_t_images'))
mask_images = [op.join(data_dir,'mask_subj%02d.nii'%n)
               for n in range(nbsubj)]

learn_images =[[ op.join(data_dir,'spmT_%04d_subj_%02d.nii'%(nb,n))
                 for nb in numbeta]
                 for n in range(nbsubj)]
test_images=learn_images

nbeta = len(numbeta)

# parameter for the intersection of the mask
ths = nbsubj/2

# possibly, dimension reduction can performed on the input data
# (not recommended)
fdim = 3

# verbosity mode
verbose = 1

# number of parcels
nbparcel = 500

# write dir
swd = tempfile.mkdtemp()


# prepare the parcel structure
fpa,ldata,Talairach = parcel_input(mask_images,nbeta,learn_images,ths,fdim)
fpa.k = nbparcel

# run the algorithm
fpa = hparcel(fpa,ldata,Talairach)
#fpa,prfx0 = hparcel(fpa,ldata,Talairach,nbperm=200,niter=5,verbose)

#produce some output images
Parcellation_output(fpa,mask_images,learn_images,Talairach,subj_id,
                    verbose=1,swd=swd)

# do some parcellation-based analysis:
# take some test images whose parcel-based signal needs to be assessed 
test_images=learn_images

# a design matrix for possibly subject-specific effects
DMtx = None

# compute and write the parcel-based statistics
Parcellation_based_analysis(fpa,test_images,numbeta,swd,DMtx,verbose)

print "Wrote everything in %s" %swd
