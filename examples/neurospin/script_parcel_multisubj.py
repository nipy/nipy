"""
Example of script to parcellate mutli-subject data
author: Bertrand Thirion, 2005-2008
"""

from fff2.spatial_models.parcellation import Parcellation
import numpy as N
import cPickle
from fff2.spatial_models.parcel_io_nii import *
from fff2.spatial_models.hierarchical_parcellation import hparcel

# number of subjects
nbru = range(1,13)
Sess = len(nbru)

# functional contrast
numbeta = [31]
nbeta = len(numbeta)

# parameter for the intersection of the mask
ths = Sess/2

# possibly, dimension reduction can perfiomedon the input data
# (not recommended)
fdim = 3

# verbosity mode
verbose = 1

# number of parcels
nbparcel = 500

# write dir
swd = "/tmp/"

# load some mask images to define the brain mask of the different subjects
Mask_Images =["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/mask.img" % bru for bru in nbru]

# load some activation images to learn the functional organization of the group of subjects
learn_images = [["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/spmT_%04d.img" % (bru, n) for n in numbeta] for bru in nbru]


# prepare the parcel structure
fpa,ldata,Talairach = parcel_input(Mask_Images,nbeta,learn_images,ths,fdim)
fpa.k = nbparcel

# run the algorithm

fpa = hparcel(fpa,ldata,Talairach)
#fpa,prfx0 = hparcel(fpa,ldata,Talairach,nbperm=200,niter=5,verbose)

#produce some output images
Parcellation_output(fpa,Mask_Images,learn_images,Talairach,nbru,verbose=1,swd = "/tmp")

# do some parcellation-based analysis:
# load some test images whose parcel-based signal needs to be assessed 
numbeta = [31]
test_images = [["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/con_%04d.img" % (bru, n) for n in numbeta] for bru in nbru]

# a design matrix for possibly subject-specific effects
DMtx = None

# compute and write the parcel-based statistics
Parcellation_based_analysis(fpa,test_images,numbeta,swd,DMtx,verbose)


