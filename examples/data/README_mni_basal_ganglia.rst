#############################################
README for ``mni_basal_ganglia.nii.gz`` image
#############################################

I extracted these basal ganglia definitions from the MNI
ICBM 2009c Nonlinear Symmetric template at 1×1x1 mm resolution.

At the time, the templates were available here:

http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009

The script to extract the data was::

    from os.path import join as pjoin
    import numpy as np

    import nibabel as nib

    atlas_fname = pjoin('mni_icbm152_nlin_sym_09c',
                        'mni_icbm152_t1_tal_nlin_sym_09a_atlas',
                        'AtlasGrey.mnc')
    atlas_img = nib.load(atlas_fname)
    # Data is in fact uint8, but with trivial float scaling
    data = atlas_img.get_data().astype(np.uint8)
    bg_data = np.zeros_like(data)
    for code in (14, 16, 39, 53): # LR striatum, LR caudate
        in_mask = data == code
        bg_data[in_mask] = code
    bg_img = nib.Nifti1Image(bg_data, atlas_img.affine)
    bg_img = nib.as_closest_canonical(bg_img)
    nib.save(bg_img, 'basal_ganglia.nii.gz')

**********
Data codes
**********

These are the values in the image:

* 14 - Left striatum
* 16 - Right striatum
* 39 - Left caudate
* 53 - Right caudate

Everything else is zero.

*******
License
*******

Contents of the file ``COPYING`` in the template archive:

Copyright (C) 1993-2004 Louis Collins, McConnell Brain
Imaging Centre, Montreal Neurological Institute, McGill University.
Permission to use, copy, modify, and distribute this software and
its documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appear in all copies.  The
authors and McGill University make no representations about the
suitability of this software for any purpose.  It is provided "as
is" without express or implied warranty.  The authors are not
responsible for any data loss, equipment damage, property loss, or
injury to subjects or patients resulting from the use or misuse of
this software package.
