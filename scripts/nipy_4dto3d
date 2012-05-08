#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Tiny script to write 4D files in any format that we read (nifti,
analyze, MINC, at the moment, as nifti 3D files '''

import os

import nipy.externals.argparse as argparse
import nipy.io.imageformats as nii


def main():
    # create the parser
    parser = argparse.ArgumentParser()
    # add the arguments
    parser.add_argument('filename', type=str,
                        help='4D image filename')
    # parse the command line
    args = parser.parse_args()
    img = nii.load(args.filename)
    imgs = nii.four_to_three(img)
    froot, ext = os.path.splitext(args.filename)
    if ext in ('.gz', '.bz2'):
        froot, ext = os.path.splitext(froot)
    for i, img3d in enumerate(imgs):
        fname3d = '%s_%04d.nii' % (froot, i)
        nii.save(img3d, fname3d)


if __name__ == '__main__':
    main()

        
