#!/usr/bin/env python
''' Analyze, plot time series difference metrics'''

import nipy.externals.argparse as argparse
import nipy.algorithms.diagnostics as nad


def main():
    # create the parser
    parser = argparse.ArgumentParser()
    # add the arguments
    parser.add_argument('filename', type=str)
    # parse the command line
    args = parser.parse_args()
    nad.plot_tsdiffs_image(args.filename)


if __name__ == '__main__':
    main()
