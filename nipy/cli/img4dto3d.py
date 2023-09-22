# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
DESCRIP = 'Read 4D image file and write 3D nifti file for each volume'
EPILOG = \
'''nipy_4dto3d will generate a series of 3D nifti images for each volume a 4D
image series in any format readable by `nibabel`.
'''
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from os.path import join as pjoin
from os.path import split as psplit
from os.path import splitext

import nibabel as nib


def main():
    parser = ArgumentParser(description=DESCRIP,
                            epilog=EPILOG,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('filename', type=str,
                        help='4D image filename')
    parser.add_argument('--out-path', type=str,
                        help='path for output image files')
    args = parser.parse_args()
    out_path = args.out_path
    img = nib.load(args.filename)
    imgs = nib.four_to_three(img)
    froot, ext = splitext(args.filename)
    if ext in ('.gz', '.bz2'):
        froot, ext = splitext(froot)
    if out_path is not None:
        pth, fname = psplit(froot)
        froot = pjoin(out_path, fname)
    for i, img3d in enumerate(imgs):
        fname3d = '%s_%04d.nii' % (froot, i)
        nib.save(img3d, fname3d)
