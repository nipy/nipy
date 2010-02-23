#!/usr/bin/env python

import os
from os.path import join as pjoin
import sys

import numpy as np

import nipy.externals.argparse as argparse
import nipy.algorithms.diagnostics as nad
from nipy.io.imageformats.filename_parser, splitext_addext

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # create the parser
    parser = argparse.ArgumentParser()
    # add the arguments
    parser.add_argument('image', type=str,
                        help='4D image filename')
    parser.add_argument('--out-root', type=str,
                        help='root for output image names')
    parser.add_argument('--ncomponents', type=int, default=10,
                        help='number of PCA components to write')
    # parse the command line
    args = parser.parse_args()
    # process inputs
    filename = args.filename
    out_root = args.out_root
    ncomps = args.ncomponents
    if out_root is None:
        out_root, ext, gz = splitext_addext(filename)
    res = nad.screen(img, ncomps)
    # save images
    for key in 'mean', 'min', 'max', 'std', 'pca':
    nii.save(mean, pjoin(pth, 'mean_' + fname))
    nii.save(std, pjoin(pth, 'std_' + fname))
    nii.save(pca, pjoin(pth, 'pca_' + fname))
    # plot, save component time courses
    plt.figure()
    for c in range(ncomp):
        plt.subplot(ncomp, 1, c+1)
        plt.plot(vectors[:,c])
    plt.savefig(pjoin(pth, 'components_%s.png' % froot))
    # plot tsdiffana
    plt.figure()
    axes = [plt.subplot(4, 1, i+1) for i in range(4)]
    nad.plot_tsdiffs(tsd, axes)
    plt.savefig(pjoin(pth, 'tsdiff_%s.png' % froot))
