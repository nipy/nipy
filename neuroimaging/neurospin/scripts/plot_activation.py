#!/usr/bin/env python
__doc__ = """%prog [options] nifti_files

Plot activation  map for all the nifti files given on the command line.
By default the activation threshold as well as the cut coordinate for the
2D views are computed automatically. The output are png files named
similarly to the input nifti or analyse files.

Warning: If you are using a version of VTK less than 5.2, a window will
appear for the rendering. Keep it on top of other windows for the 3D view
to saved properly. """


# Author: Gael Varoquaux
# License: BSD

# standard library imports
from optparse import OptionParser
import os
import sys

# local imports
from fff2.viz.activation_maps import plot_niftifile, SformError, \
        NiftiIndexError


parser = OptionParser(__doc__)
parser.add_option("-o", "--outdir", dest="out_dir",
                  help="write all output to DIR", metavar="DIR")
parser.add_option("-f", "--htmlfile", dest="html_file",
                  help="write report to a html file FILE, useful when "
                  "visualizing multiple files", metavar="FILE")
parser.add_option("-a", "--anat", dest="anat",
                  help="use the given file as an anatomy", metavar="FILE")
parser.add_option("-M", "--mask", dest="mask",
                  help="use the given file as a mask", metavar="FILE")
parser.add_option("-d", "--no3d",
                  action="store_false", dest="do3d", default=True,
                  help="don't try to do a 3D view")
parser.add_option("-s", "--sign",
                  action="store_false", dest="auto_sign", default=True,
                  help="force activation sign to be positive")
parser.add_option('-c', '--cut-coords', dest="cut_coords", 
                  type="float", nargs=3, 
                  help="Talairach coordinates of the 2D cuts")
parser.add_option('-m', '--vmin', dest='vmin',
                  type='float',
                  help='Minimum value for the activation, used for '
                  'thresholding')

def relative_path(root_path, path):
    """ Returns the path relative to 'root_path'

        >>> relative_path('../utils', '../utils/mask.py')
        mask.py
        >>> relative_path('../foobar', '../utils/mask.py')
        ../utils/mask.py
    """
    root_path = os.path.abspath(root_path)
    path = os.path.abspath(path)
    prefix = os.path.commonprefix((root_path, path))
    root_path = root_path[len(prefix):]
    path = path[len(prefix):]
    root_path = [a for a in root_path.split('/') if a]
    path = [a for a in path.split('/') if a]
    return '/'.join(['..']*len(root_path) + path)

def main(argv=None):
    """ Main entry point. Provide argv, a list of arguments, to override
        sys.argv.
    """
    try:
        import pylab
    except Exception, e:
        print >> sys.stderr, "You do not seem to have a working install"\
            " of matplotlib: "
        print e
    options, args = parser.parse_args(argv)
    if len(args) == 0:
        parser.print_help()
        return 0
    error_no = 0
    if options.anat and not os.path.exists(options.anat):
        print >>sys.stderr, 'Anatomy file %s not found' % options.anat
        return 2
    if options.mask and not os.path.exists(options.mask):
        print >>sys.stderr, 'Mask file %s not found' % options.mask
        return 2


    output_list = list()

    for filename in args:
        if not os.path.exists(filename):
            error_no = 1
            print >>sys.stderr, 'File %s not found' % filename
        outputname = os.path.abspath(os.path.splitext(filename)[0])
        if outputname.endswith('.nii'):
            # Cater for .nii.gz
            outputname = os.path.splitext(outputname)[0]
        if options.out_dir:
            outputname = os.path.join(
                            options.out_dir,
                            os.path.basename(outputname)
                    )
        outputname = outputname + '.png'
        print 'Rendering file %s to %s' % (filename, outputname)
        try:
            if options.do3d:
                do3d = 'offscreen'
            else:
                do3d = False
            output_list.extend(
                plot_niftifile(filename, do3d=do3d, vmin=options.vmin,
                       cut_coords=options.cut_coords, 
                       anat_filename=options.anat,
                       mask_filename=options.mask,
                       outputname=outputname,
                       figure_num=1,
                       auto_sign=options.auto_sign,
                        )
                )
            pylab.clf()
        except SformError:
            print 'file: %s sform is not inversible' % filename
            error_no = 3
        except NiftiIndexError, e:
            print e

    if options.html_file:
        html_file = file(options.html_file, 'w')
        html_file.write('<html>\n<body>\n')
        for filename in output_list:
            image_path = relative_path(os.path.dirname(options.html_file), 
                                       filename)
            html_file.write("""
<p style="display: inline ;">
<span style="display: table-cell">
&nbsp;
<a href="%s" style="border: 0pt;">
<img style="vertical-align: middle ; width: 40em; border: 0pt;" 
     src="%s">
</a>
</span>
</p>
""" % (image_path, image_path))
        html_file.write('</body>\n</html>')
        print "Wrote report to %s" % options.html_file

    return error_no

if __name__ == '__main__':
    main()


