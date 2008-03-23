"""A repository of data used for doctests, examples and testing.

The neuroimaging.data directory will contain a small set of imaging data 
to be used in example scripts, doctests, and unittests.  The data files 
should be of 'gold standard' quality, or at least of known origins by the 
nipy development team.

By using one data directory, all developers and users will know exactly
where to look for the imaging files.  Keeping data files in one location
will also reduce the number of path/file references throughout the codebase.

This dataset should be relatively small, but contain enough files to provide
reasonable testing and a set of files for users to get started with nipy.

Examples
--------

>>> from neuroimaging.data import MNI_file
>>> from neuroimaging.core.image import image
>>> img = image.load(MNI_file)
>>> img.shape
(91, 109, 91)

"""

import os

__all__ = ['MNI_file', 'avganat_file']

# Discover directory path
filepath = os.path.abspath(__file__)
basedir = os.path.dirname(filepath)

# NOTE:  Consider using gzip/bzip2 files and DataSource for opening

# Initialize paths to standard data files
# This file was copied from an FSL install.  Most likely it's in Radiological coordinates!
MNI_file = os.path.join(basedir, 'MNI152_T1_2mm.nii')

# A FIAC file.  Full origins unknown.
avganat_file = os.path.join(basedir, 'avganat.img')
