# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Command line wrapper of SpaceTimeRealign

Based on:

Alexis Roche (2011) A Four-Dimensional Registration Algorithm With Application
to Joint Correction of Motion and Slice Timing in fMRI. IEEE Trans. Med.
Imaging 30(8): 1546-1554
"""

import argparse
import os.path as op

import nipy.algorithms.registration as reg

parser = argparse.ArgumentParser()

parser.add_argument('TR', type=float, metavar='Float', help="""The TR of the measurement""")

parser.add_argument('input', type=str, metavar='File',
                help="""Path to a nifti file, or to a folder containing nifti files. If a path to a folder is provided, the order of motion correction will be np.sort(list_of_files). The outputs will be '*_mc.par' (containing 3 translation and three rotation parameters) and '*_mc.nii.gz' containing the motion corrected data (unless 'apply' is set to False)""")

parser.add_argument('--slice_order', type=str, metavar='String',
                    help="""The order of slice acquisition {'ascending', 'descending' (default), or the name of a function from `nipy.algorithms.slicetiming.timefuncs`}""", default='descending')

parser.add_argument('--slice_dim', type=int, metavar='Int', help="""Integer
denoting the axis in `images` that is the slice axis.  In a 4D image, this will
often be axis = 2 (default).""", default=2)

parser.add_argument('--slice_dir', type=int, metavar='Int', help=""" 1 if the
slices were acquired slice 0 first (default), slice -1 last, or -1 if acquire slice -1 first, slice 0 last.""", default=1)

parser.add_argument('--make_figure', type=bool, metavar='Bool',
                help="""Whether to generate a '.png' figure with the motion parameters across runs. {True, False}. Default: False """, default=False)

parser.add_argument('--save_path', type=str, metavar='String',
                    help="""Full path to a file-system location for the output files. Defaults to the same location as the input files""",
                    default='none')

parser.add_argument('--save_params', type=bool, metavar='Bool',
                help="""Whether to save the motion corrections parameters (3 rotations, 3 translations). {True, False}. Default: False. NOTE: The rotations are not Euler angles, but a rotation vector. Use `nipy.algorithms.registration.to_matrix44` to convert to a 4-by-4 affine matrix""", default=False)


def main():
    args = parser.parse_args()

    if args.save_path == 'none':
        save_path = op.split(args.input)[0]
    else:
        save_path = args.save_path

    xform = reg.space_time_realign(args.input, float(args.TR),
                                   slice_order=args.slice_order,
                                   slice_dim=int(args.slice_dim),
                                   slice_dir=int(args.slice_dir),
                                   # We always apply the xform in the cli
                                   apply=True,
                                   make_figure=args.make_figure,
                                   out_name=save_path)

    if args.save_params:
        f = open(op.join(save_path, 'mc.par'), 'w')
        for x in xform:
            euler_rot = reg.aff2euler(x.as_affine())
            for r in euler_rot:
                f.write(f'{r}\t')
            for t in x.translation[:-1]:
                f.write(f'{t}\t')
            f.write(f'{x.translation[-1]}\n')
        f.close()
