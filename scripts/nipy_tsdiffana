#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Analyze, plot time series difference metrics'''

import nipy.externals.argparse as argparse
import nipy.algorithms.diagnostics as nad


def main():
    # create the parser
    parser = argparse.ArgumentParser()
    # add the arguments
    parser.add_argument('--out-file', type=str,
                        help='graphics file to write to instead '
                        'of leaving image on screen')
    parser.add_argument('filename', type=str,
                        help='4D image filename')
    # parse the command line
    args = parser.parse_args()
    show = args.out_file is None
    axes = nad.plot_tsdiffs_image(args.filename, show=show)
    if args.out_file:
        axes[0].figure.savefig(args.out_file)


if __name__ == '__main__':
    main()
