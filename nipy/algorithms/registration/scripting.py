#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
A scripting wrapper around 4D registration (SpaceTimeRealign)
"""

import os
import os.path as op
import numpy as np
import numpy.linalg as npl

import nibabel as nib
import nibabel.eulerangles as euler
from nibabel.optpkg import optional_package
matplotlib, HAVE_MPL, _ = optional_package('matplotlib')

if HAVE_MPL:
    import matplotlib.pyplot as plt

from .groupwise_registration import SpaceTimeRealign
import nipy.externals.argparse as argparse
import nipy.algorithms.slicetiming as st
timefuncs = st.timefuncs.SLICETIME_FUNCTIONS

__all__ = ["space_time_realign", "aff2euler"]


def aff2euler(affine):
    """
    Compute Euler angles from 4 x 4 `affine`

    Parameters
    ----------
    affine : 4 by 4 array
        An affine transformation matrix

    Returns
    -------
    The Euler angles associated with the affine
    """
    return euler.mat2euler(aff2rot_zooms(affine)[0])


def aff2rot_zooms(affine):
    """
    Compute a rotation matrix and zooms from 4 x 4 `affine`

    Parameters
    ----------
    affine : 4 by 4 array
        An affine transformation matrix

    Returns
    -------
    R: 3 by 3 array
       A rotation matrix in 3D

    zooms: length 3 1-d array
         Vector with voxel sizes.
    """
    RZS = affine[:3, :3]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    RS = RZS / zooms
    # Adjust zooms to make RS correspond (below) to a true
    # rotation matrix.
    if npl.det(RS) < 0:
        zooms[0] *= -1
        RS[:,0] *= -1
    # retrieve rotation matrix from RS with polar decomposition.
    # Discard shears
    P, S, Qs = npl.svd(RS)
    R = np.dot(P, Qs)
    return R, zooms


def space_time_realign(input, tr, slice_order='descending', slice_dim=2,
                       slice_dir=1, apply=True, make_figure=False,
                       out_name=None):
    """
    This is a scripting interface to `nipy.algorithms.registration.SpaceTimeRealign` 

    Parameters
    ----------
    input : str or list
        A full path to a file-name (4D nifti time-series) , or to a directory
        containing 4D nifti time-series, or a list of full-paths to files.
    tr : float
        The repetition time
    slice_order : str (optional)
        This is the order of slice-times in the acquisition. This is used as a
        key into the ``SLICETIME_FUNCTIONS`` dictionary from
        :mod:`nipy.algorithms.slicetiming.timefuncs`. Default: 'descending'.
    slice_dim : int (optional) 
        Denotes the axis in `images` that is the slice axis.  In a 4D image,
        this will often be axis = 2 (default).
    slice_dir : int (optional)
        1 if the slices were acquired slice 0 first (default), slice -1 last,
        or -1 if acquire slice -1 first, slice 0 last.
    apply : bool (optional)
        Whether to apply the transformation and produce an output. Default:
        True. 
    make_figure : bool (optional)
        Whether to generate a .png figure with the parameters across scans.
    out_name : bool (optional)
        Specify an output location (full path) for the files that are
        generated. Default: generate files in the path of the inputs (with an
        `_mc` suffix added to the file-names.
    

    Returns
    -------
    transforms : ndarray
        An (n_times_points,) shaped array containing 
       `nipy.algorithms.registration.affine.Rigid` class instances for each time
        point in the time-series. These can be used as affine transforms by
        referring to their `.as_affine` attribute.
    """
    if make_figure:
        if not HAVE_MPL:
            e_s ="You need to have matplotlib installed to run this function"
            e_s += " with `make_figure` set to `True`"
            raise RuntimeError(e_s)
        
    # If we got only a single file, we motion correct that one:
    if op.isfile(input):
        if not (input.endswith('.nii') or input.endswith('.nii.gz')):
            e_s = "Input needs to be a nifti file ('.nii' or '.nii.gz'"
            raise ValueError(e_s)
        fnames = [input]
        input = nib.load(input)
    # If this is a full-path to a directory containing files, it's still a
    # string: 
    elif isinstance(input, str):
        list_of_files = os.listdir(input)
        fnames = [op.join(input, f) for f in np.sort(list_of_files)
                  if (f.endswith('.nii') or f.endswith('.nii.gz')) ]
        input = [nib.load(x) for x in fnames]
    # Assume that it's a list of full-paths to files:
    else:
       input = [nib.load(x) for x in input]

    slice_times = timefuncs[slice_order]
    slice_info = [slice_dim,
                  slice_dir]
    
    reggy = SpaceTimeRealign(input,
                             tr,
                             slice_times,
                             slice_info)

    reggy.estimate(align_runs=True)

    # We now have the transformation parameters in here:
    transforms = np.squeeze(np.array(reggy._transforms))
    rot = np.array([t.rotation for t in transforms])
    trans = np.array([t.translation for t in transforms])

    if apply:
        new_reggy = reggy.resample(align_runs=True)
        for run_idx, new_im in enumerate(new_reggy):
            new_data = new_im.get_data()
            # We use the top 4 by 4 as the affine for the new file we will
            # create:
            new_aff = new_im.affine[:4, :4]
            new_ni = nib.Nifti1Image(new_data, new_aff)
            # Save it out to a '.nii.gz' file:
            old_fname_split = op.split(fnames[run_idx])
            # We retain the file-name adding '_mc' regardless of where it's saved
            new_fname = old_fname_split[1].split('.')[0] + '_mc.nii.gz'
            if out_name is None:
                new_path = old_fname_split[0]                               
            else:
                new_path = out_name
            new_ni.to_filename(op.join(new_path, new_fname))

    if make_figure:
        figure, ax = plt.subplots(2)
        figure.set_size_inches([8, 6])
        ax[0].plot(rot)
        ax[0].set_xlabel('Time (TR)')
        ax[0].set_ylabel('Translation (mm)')
        ax[1].plot(trans)
        ax[1].set_xlabel('Time (TR)')
        ax[1].set_ylabel('Rotation (radians)')
        figure.savefig(op.join(os.path.split(fnames[0])[0],
                                    'mc_params.png'))

    return transforms
