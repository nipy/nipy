#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
A scripting wrapper around 4D registration (SpaceTimeRealign)

"""

import os
import numpy as np

import nibabel as nib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import nipy.algorithms.registration as reg
import nipy.externals.argparse as argparse
import nipy.algorithms.slicetiming as st
timefuncs = st.timefuncs.SLICETIME_FUNCTIONS

def space_time_realign(input, tr, slice_order=timefuncs['descending'],
                       slice_dim=2, slice_dir=1, apply=False, make_figure=False):
    """
    Scripting interface to ``
    """
    # If we got only a single file, we motion correct that one:
    if os.path.isfile(input):
        if not (input.endswith('.nii') or input.endswith('.nii.gz')):
            e_s = "Input needs to be a nifti file ('.nii' or '.nii.gz'"
            raise ValueError(e_s)
        input = nib.load(input)
        fnames = [input]
    else:
        list_of_files = os.listdir(input)
        fnames = [os.path.join(input, f) for f in np.sort(list_of_files)
                  if (f.endswith('.nii') or f.endswith('.nii.gz')) ]
        input = [nib.load(x) for x in fnames]

    slice_times = timefuncs[slice_order]
    slice_info = [slice_dim,
                  slice_dir]

    reggy = reg.SpaceTimeRealign(input,
                                 TR,
                                 slice_times,
                                 slice_info)

    reggy.estimate(align_runs=True)
    # This will be the dtype for the recarray of params in all runs:
    dt = [('t1',float), ('t2', float), ('t3', float),
          ('r1',float), ('r2', float), ('r3', float)]

    # We now have the transformation parameters in here:
    transforms = reggy._within_run_transforms

    # We'll keep the last transformation in each run, so that we can
    # concatenate the transformations between runs:
    rot_last = [0, 0, 0]
    trans_last = [0, 0, 0]

    params_arr = []

    # There's a list for every run:
    for run_idx, this_trans in enumerate(transforms):
        this_rot = np.array([t.rotation for t in this_trans])
        this_trans = np.array([t.translation for t in this_trans])
        # To create a recarray, we need to first make a list of tuples:
        prep_arr = [(this_trans[i, 0] + trans_last[0],
                     this_trans[i, 1] + trans_last[1],
                     this_trans[i, 2] + trans_last[2],
                     this_rot[i, 0] + rot_last[0],
                     this_rot[i, 1] + rot_last[1],
                     this_rot[i, 2] + rot_last[2])
                     for i in range(this_rot.shape[0])]

        params_arr.append(np.array(prep_arr, dtype=dt))
        mlab.rec2csv(params_arr[-1], fnames[run_idx].split('.')[0] + '_mc.par')

        rot_last = [prep_arr[-1][0], prep_arr[-1][1], prep_arr[-1][2]]
        trans_last = [prep_arr[-1][3], prep_arr[-1][4], prep_arr[-1][5]]

    if apply:
        # dbg
        # print("Applying motion correction")
        new_reggy = reggy.resample(align_runs=True)
        # dbg
        for run_idx, new_im in enumerate(new_reggy):
            new_data = new_im.get_data()
            # We use the top 4 by 4 as the affine for the new file we will
            # create:
            new_aff = new_im.affine[:4, :4]
            new_ni = nib.Nifti1Image(new_data, new_aff)
            # Save it out to a '.nii.gz' file:
            new_ni.to_filename(fnames[run_idx].split('.')[0] + '_mc.nii.gz')

    if make_figure:
        all_params = np.concatenate(params_arr)
        figure, ax = plt.subplots(2)
        figure.set_size_inches([8, 6])
        ax[0].plot(all_params['t1'])
        ax[0].plot(all_params['t2'])
        ax[0].plot(all_params['t3'])
        ax[0].set_xlabel('Time (TR)')
        ax[0].set_ylabel('Translation (mm)')
        ax[1].plot(all_params['r1'])
        ax[1].plot(all_params['r2'])
        ax[1].plot(all_params['r3'])
        ax[1].set_xlabel('Time (TR)')
        ax[1].set_ylabel('Rotation (radians)')
        figure.savefig(os.path.join(os.path.split(fnames[0])[0],
                                    'mc_params.png'))
