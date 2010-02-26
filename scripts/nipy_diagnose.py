#!/usr/bin/env python
''' Script to calculate and write results for diagnostic screen

nipy_diagnose will generate a series of diagnostic images for a 4D
fMRI image volume.  The following images will be generated:

    * components_<prefix>.png : plots of PCA basis vectors
    * max_<prefix>.nii : max image
    * mean_<prefix>.nii : mean image
    * min_<prefix>.nii : min image
    * pca_<prefix>.nii : 4D image of PCA component images
    * pcnt_var_<prefix>.png : XXX
    * std_<prefix>.nii : standard deviation image XXX
    * tsdiff_<prefix>.png : XXX

The generated files will be saved in the directory specified by the
--out-path parameter.  If the out-path directory is not specified,
generated images will be saved in the same directory as the input
image.

'''
import os

import numpy as np

import nipy
import nipy.externals.argparse as argparse
import nipy.algorithms.diagnostics.screens as nads
from nipy.io.imageformats.filename_parser import splitext_addext

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    # create the parser
    desc = __doc__.splitlines()[0]
    parser = argparse.ArgumentParser(description=desc)
    # add the arguments
    parser.add_argument('filename', type=str,
                        help='4D image filename')
    parser.add_argument('--out-path', type=str,
                        help='path for output image files')
    parser.add_argument('--out-fname-prefix', type=str,
                        help='prefix for output image names')
    parser.add_argument('--ncomponents', type=int, default=10,
                        help='number of PCA components to write')
    parser.add_argument('--extended-help', action='store_true',
                        help='print extended help')

    # HACK: Allow user to get additional help without having to
    # provide a filename (or any positional arg).  Could not find a
    # better way to do this with argparse.
    import sys
    if '--extended-help' in sys.argv:
        print __doc__
        return

    # parse the command line
    args = parser.parse_args()
    # process inputs
    filename = args.filename
    out_path = args.out_path
    out_root = args.out_fname_prefix
    ncomps = args.ncomponents
    # collect extension for output images
    froot, ext, gz = splitext_addext(filename)
    pth, fname = os.path.split(froot)
    if out_path is None:
        out_path = pth
    if out_root is None:
        out_root = fname
    img = nipy.load_image(filename)
    res = nads.screen(img, ncomps)
    nads.write_screen_res(res, out_path, out_root, ext + gz)
    
if __name__ == '__main__':
    main()
