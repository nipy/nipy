#!/usr/bin/env python
''' Script to calculate and write results for diagnostic screen
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


if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser()
    # add the arguments
    parser.add_argument('filename', type=str,
                        help='4D image filename')
    parser.add_argument('--out-path', type=str,
                        help='path for output image names')
    parser.add_argument('--out-fname-prefix', type=str,
                        help='prefix for output image names')
    parser.add_argument('--ncomponents', type=int, default=10,
                        help='number of PCA components to write')
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
    
