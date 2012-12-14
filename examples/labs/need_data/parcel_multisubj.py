#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
"""
Example of script to parcellate mutli-subject data.

May take some time to complete.

Author: Bertrand Thirion, 2005-2009
"""

from os import mkdir, getcwd, path

from numpy import array

from nipy.labs.spatial_models.parcel_io import parcel_input, \
    write_parcellation_images, parcellation_based_analysis
from nipy.labs.spatial_models.hierarchical_parcellation import hparcel

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset

# Get the data
nb_subj = 12
subj_id = ['subj_%02d' % s for s in range(nb_subj)]
nbeta = '0029'
data_dir = path.join(DATA_DIR, 'group_t_images')
mask_images = [path.join(data_dir, 'mask_subj%02d.nii' % n)
               for n in range(nb_subj)]

learn_images = [path.join(data_dir, 'spmT_%s_subj_%02d.nii' % (nbeta, n))
                for n in range(nb_subj)]
missing_file = array(
    [not path.exists(m) for m in mask_images + learn_images]).any()
learn_images = [[m] for m in learn_images]

if missing_file:
    get_second_level_dataset()

# parameter for the intersection of the mask
ths = .5

# number of parcels
nbparcel = 200

# write directory
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

# prepare the parcel structure
domain, ldata = parcel_input(mask_images, learn_images, ths)

# run the algorithm
fpa = hparcel(domain, ldata, nbparcel, verbose=1)

# produce some output images
write_parcellation_images(fpa, subject_id=subj_id, swd=write_dir)

# do some parcellation-based analysis:
# take some test images whose parcel-based signal needs to be assessed
test_images = [path.join(data_dir, 'spmT_%s_subj_%02d.nii' % (nbeta, n))
               for n in range(nb_subj)]

# compute and write the parcel-based statistics
rfx_path = path.join(write_dir, 'prfx_%s.nii' % nbeta)
parcellation_based_analysis(fpa, test_images, 'one_sample', rfx_path=rfx_path)
print("Wrote everything in %s" % write_dir)
