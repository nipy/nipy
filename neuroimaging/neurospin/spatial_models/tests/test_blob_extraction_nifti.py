"""
This scipt takes a (nifti) image and extarcts the blobs from it.
the output is
1) a label image whioch delineates the blobs
2) a continuous-values image that yields the average signal per blob

simply modify the input image path to make it work on your preferred
nifti image

Author : Bertrand Thirion, 2008-2009
"""
#autoindent

# Standard library imports
import os.path as op

# Third party imports
import nifti
import nose.tools as nt
import numpy as np
import numpy.testing.decorators as dec
import scipy.stats as st

# Our own imports
from neuroimaging.neurospin.graph import field as ff


# This isn't really a standalone test, so mark it as such for nose.  Eventually
# it will be hopefully refactored into a proper test.
@nt.nottest
def main():
    # Get the data
    nbru = range(1,13)
    nbeta = [29]
    swd = "/tmp/"

    # a mask of the brain in each subject
    Mask_Images =["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/mask.img" % bru for bru in nbru]

    # activation image in each subject
    betas = [["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/spmT_%04d.img" % (bru, n) for n in nbeta] for bru in nbru]
    #
    s=6

    # Read the masks
    rmask = nifti.NiftiImage(Mask_Images[s])
    ref_dim = rmask.getVolumeExtent()
    mask = (rmask.data).T

    xyz = np.array(np.where(mask))
    nbvox = np.size(xyz,1)

    # Read the data
    rbeta = nifti.NiftiImage(betas[s][0])
    beta = (rbeta.data).T
    beta = beta[mask!=0]

    # build the field
    F = ff.Field(nbvox)
    F.from_3d_grid(np.transpose(xyz),18)
    F.set_field(np.reshape(beta,(nbvox,1)))

    # compute the blobs
    th = float(st.t.isf(0.01,100))
    smin = 5
    nroi = F.generate_blobs(refdim=0,th=th,smin = smin)

    # compute the average signal within each blob
    idx = nroi.get_seed()
    parent = nroi.get_parent()
    label = nroi.get_label()
    nroi.make_feature(beta, 'height','mean')
    Bfm = nroi.get_ROI_feature('height')

    #write the resulting blob and signal-per-blob images
    Label = -np.ones(ref_dim,'i')
    Label[mask!=0] = label
    nim = nifti.NiftiImage(Label.T,rbeta.header)
    nim.description = "Blob image"
    LabelImage = op.join(swd,"blob.nii")
    nim.save(LabelImage)

    bmap = np.zeros(ref_dim)
    if nroi.k>0: bmap[mask!=0] = Bfm[label]*(label>-1)
    nim = nifti.NiftiImage(bmap.T,rbeta.header)
    nim.description = "Blob-average activation image"
    LabelImage = op.join(swd,"bmap.nii")
    nim.save(LabelImage)


# If run as a script, call main
if __name__ == '__main__':
    main()
