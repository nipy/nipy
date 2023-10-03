# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
DESCRIP = ' Analyze, plot time series difference metrics'
EPILOG = \
'''Runs the time series difference algorithm over a 4D image volume, often an
FMRI volume.

It works in one of three modes:

* interactive : the time series difference plot appears on screen.  This is the
  default mode
* non-interactive, plot only : write time series difference plot to graphic
  file. Use the "--out-file=<myfilename>" option to activate this mode
* non-interactive, write plot, images and variables : write plot to file, and
  write generated diagnostic images and variables to files as well. Use the
  "--write-results" flag to activate this option. The generated filenames come
  from the results of the "--out-path" and "--out-fname-label" options (see
  help).

Write-results option, generated files
-------------------------------------

When doing the time point analysis, we will make a difference volume between
each time point and the next time point in the series.  If we have T volumes
then there will be (T-1) difference volumes. Call the vector of difference
volumes DV and the first difference volume DV[0].  So DV[0] results from
subtraction of the second volume in the 4D input image from the first volume in
the 4D input image.  The element-wise squared values from DV[0] is *DV2[0]*.

The following images will be generated. <ext> is the input filename extension
(e.g. '.nii'):

* "dv2_max_<label><ext>" : 3D image volume, where each slice S is slice from
  all of DV2[0] (slice S) throudh DV2[T-1] (slice S) that has the maximum
  summed squared values.  This volume gives an idea of the worst (highest
  difference) slices across the whole time series.
* "dv2_mean_<label><ext>" : the mean of all DV2 volumes DV2[0] .. DV[T-1]
  across the volume (time) dimension.  Higher voxel values in this volume mean
  that time-point to time point differences tended to be high in this voxel.

We also write the mean signal at each time point, and the mean squared
difference between each slice in time, as variables to a 'npz' file named
"tsdiff_<label>.npz"

The filenames for the outputs are of the form
<out-path>/<some_prefix><label><file-ext> where <out-path> is the path
specified by the --out-path option, or the path of the input filename;
<some_prefix> is one of the standard prefixes above, <label> is given by
--out-label, or by the filename of the input image (with path and extension
removed), and <file-ext> is '.png' for graphics, or the extension of the input
filename for volume images.  For example, specifying only the input filename
``/some/path/fname.img`` will generate filenames of the form
``/some/path/tsdiff_fname.png, /some/path/dv2_max_fname.img`` etc.
'''

from argparse import ArgumentParser, RawDescriptionHelpFormatter


def main():
    parser = ArgumentParser(description=DESCRIP,
                            epilog=EPILOG,
                            formatter_class=RawDescriptionHelpFormatter)
    # add the arguments
    parser.add_argument('filename', type=str,
                        help='4D image filename')
    parser.add_argument('--out-file', type=str,
                        help='graphics file to write to instead '
                        'of leaving image on screen')
    parser.add_argument('--write-results', action='store_true',
                        help='if specified, write diagnostic images and '
                        'analysis variables, plot to OUT_PATH. Mutually '
                        'incompatible with OUT_FILE')
    parser.add_argument('--out-path', type=str,
                        help='path for output image files (default from '
                        'FILENAME path')
    parser.add_argument('--out-fname-label', type=str,
                        help='mid part of output image / plot filenames')
    parser.add_argument('--time-axis', type=str, default='t',
                        help='Image axis for time')
    parser.add_argument('--slice-axis', type=str, default=None,
                        help='Image axis for slice')
    # parse the command line
    args = parser.parse_args()
    if args.out_file is not None and args.write_results == True:
        raise RuntimeError("OUT_FILE option not compatible with WRITE_RESULTS"
                           " option")
    show = args.out_file is None and args.write_results == False
    if not show:
        import matplotlib
        matplotlib.use('Agg')
    else:
        import matplotlib.pyplot as plt
    # Import late to give set of mpl backend best chance of working
    from nipy.algorithms.diagnostics.commands import tsdiffana
    tsdiffana(args)
    if show:
        plt.show()
