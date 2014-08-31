# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Example running a parcel-based second-level analysis from a set of
first-level effect images.

This script takes as input a directory path that contains first-level
images in nifti format, as well as a group mask image and a
parcellation image (such as the AAL atlas, 'ROI_MNI_V4.nii', see
http://www.gin.cnrs.fr/spip.php?article217). All images are assumed to
be in a common reference space, e.g. the MNI/Talairach space.

It outputs three images:

* tmap.nii.gz, a `t-statistic` image similar to a SPM-like second-level
  t-map, except it is derived under an assumption of localization
  uncertainty in reference space.

* parcel_mu.nii.gz, an image that maps each voxel to the estimated
  population effect in the parcel it belongs to.

* parcel_prob.nii.gz, an image that maps each voxel to the probability
  that the population effect in the parcel it belongs to is
  positive-valued.

See the `nipy.algorithms.group.ParcelAnalysis` class for more general
usage information.
"""
from os.path import join
from glob import glob
from nipy import load_image
from nipy.algorithms.group import parcel_analysis
from nipy.externals.argparse import ArgumentParser

# Parse command line
description = 'Run a parcel-based second-level analysis from a set of\
first-level effect images.'

parser = ArgumentParser(description=description)
parser.add_argument('con_path', metavar='con_path',
                    help='directory where 1st-level images are to be found')
parser.add_argument('msk_file', metavar='msk_file',
                    help='group mask file')
parser.add_argument('parcel_file', metavar='parcel_file',
                    help='parcellation image file')
args = parser.parse_args()

# Load first-level images
con_files = glob(join(args.con_path, '*.nii'))
con_imgs = [load_image(f) for f in con_files]

# Load group mask
msk_img = load_image(args.msk_file)

# Load parcellation
parcel_img = load_image(args.parcel_file)

# Run parcel analysis and write output images in the current directory
effect_img, proba_img = parcel_analysis(con_imgs, parcel_img,
                                        msk_img=msk_img, fwhm=8,
                                        res_path='.')
