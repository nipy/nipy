#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Tiny script to write 4D file in any format that we write (nifti,
analyze, at the moment, from input 3d files '''

import os
from os.path import join as pjoin

import nipy.externals.argparse as argparse
import nipy.io.imageformats as nii


def do_3d_to_4d(filenames, check_affines=True):
    imgs = []
    for fname in filenames:
        img = nii.load(fname)
        imgs.append(img)
    return nii.concat_images(imgs, check_affines=check_affines)


def main():
    # create the parser
    parser = argparse.ArgumentParser()
    # add the arguments
    parser.add_argument('in_filenames', type=str,
                        nargs='+', 
                        help='3D image filenames')
    parser.add_argument('--out-4d', type=str,
                        help='4D output image name')
    parser.add_argument('--check-affines', type=bool,
                        default=True,
                        help='False if you want to ignore differences '
                        'in affines between the 3D images, True if you '
                        'want to raise an error for significant '
                        'differences (default is True)')
    # parse the command line
    args = parser.parse_args()
    # get input 3ds
    filenames = args.in_filenames
    # affine check
    check_affines = args.check_affines
    # get output name
    out_fname = args.out_4d
    if out_fname is None:
        pth, fname = os.path.split(filenames[0])
        froot, ext = os.path.splitext(fname)
        if ext in ('.gz', '.bz2'):
            gz = ext
            froot, ext = os.path.splitext(froot)
        else:
            gz = ''
        out_fname = pjoin(pth, froot + '_4d' + ext + gz)
    img4d = do_3d_to_4d(filenames, check_affines=check_affines)
    nii.save(img4d, out_fname)
    

if __name__ == '__main__':
    main()

        
