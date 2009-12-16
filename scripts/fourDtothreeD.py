#!/usr/bin/env python
''' Tiny script to write 4D files in any format that we read (nifti,
analyze, MINC, at the moment, as nifti 3D files '''

import os
import sys

import nipy.io.imageformats as nii


if __name__ == '__main__':
    try:
        fname = sys.argv[1]
    except IndexError:
        raise OSError('Expecting 4d image filename')
    img = nii.load(fname)
    imgs = nii.four_to_three(img)
    froot, ext = os.path.splitext(fname)
    if ext in ('.gz', '.bz2'):
        froot, ext = os.path.splitext(froot)
    for i, img3d in enumerate(imgs):
        fname3d = '%s_%04d.nii' % (froot, i)
        nii.save(img3d, fname3d)
        
