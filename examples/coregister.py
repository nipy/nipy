"""Example using Tom's registration code from scipy.

"""

from os import path
from glob import glob

import scipy.ndimage._registration as reg

# Data files
basedir = '/Users/cburns/data/twaite'
anatfile = path.join(basedir, 'ANAT1_V0001.img')
funcdir = path.join(basedir, 'fMRIData')
fileglob = path.join(funcdir, 'FUNC1_V000?.img')   # Get first 10 images

if __name__ == '__main__':
    print 'Coregister anatomical:\n', anatfile
    print '\nWith these functional images:'
    funclist = glob(fileglob)
    for func in funclist:
        print func
    measures, imageF_anat, fmri_series = \
    reg.demo_MRI_coregistration(anatfile, funclist[0:4])
