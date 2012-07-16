# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Get two images from the web (one mask image and one spmT image) and put them in
the nipy user dir - usually therefore at ``~/.nipy/tests/data``.

Author : Bertrand Thirion, 2009
"""

import os
import urllib2
import tarfile

from nibabel.data import get_nipy_user_dir

NIPY_DIR = get_nipy_user_dir()
DATA_DIR = os.path.join(NIPY_DIR, 'tests', 'data')

def get_second_level_dataset():
    """ Lightweight dataset for multi-subject analysis
    """
    # define several paths
    url = 'ftp://ftp.cea.fr/pub/dsv/madic/download/nipy'
    mask_image = os.path.join(DATA_DIR, 'mask.nii.gz')
    input_image = os.path.join(DATA_DIR, 'spmT_0029.nii.gz')
    group_data = os.path.join(DATA_DIR, 'group_t_images.tar.gz')

    # if needed create DATA_DIR
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        assert os.path.exists(DATA_DIR)

    # download mask_image if necessary
    if not os.path.exists(mask_image):
        filename = 'mask.nii.gz'
        datafile = os.path.join(url, filename)
        fp = urllib2.urlopen(datafile)
        local_file = open(mask_image, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    # download input_image if necessary
    if not os.path.exists(input_image):
        filename = 'spmT_0029.nii.gz'
        datafile = os.path.join(url, filename)
        fp = urllib2.urlopen(datafile)
        local_file = open(input_image, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    # download group_data if necessary
    if not os.path.exists(group_data):
        filename = 'group_t_images.tar.gz'
        datafile = os.path.join(url, filename)
        fp = urllib2.urlopen(datafile)
        local_file = open(group_data, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    # untargzip group_data
    tar = tarfile.open(group_data)
    tar.extractall(DATA_DIR)
    tar.close()
    os.remove(group_data)
    return DATA_DIR


def get_first_level_dataset():
    """ Heavier dataset (30 MO) for first-level analysis
    """
    # define several paths
    url = 'ftp://ftp.cea.fr/pub/dsv/madic/download/nipy'
    raw_fmri = os.path.join(DATA_DIR, 's12069_swaloc1_corr.nii.gz')
    paradigm = os.path.join(DATA_DIR, 'localizer_paradigm.csv')

    # create DATA_DIR
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        assert os.path.exists(DATA_DIR)

    # download mask_image if necessary
    if not os.path.exists(paradigm):
        print 'Downloading mask image, this may take time'
        datafile = os.path.join(url, 'localizer_paradigm.csv')
        fp = urllib2.urlopen(datafile)
        local_file = open(paradigm, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    # download raw_fmri if necessary
    if not os.path.exists(raw_fmri):
        print 'Downloading fmri image, this may take time'
        filename = 's12069_swaloc1_corr.nii.gz'
        datafile = os.path.join(url, filename)
        fp = urllib2.urlopen(datafile)
        local_file = open(raw_fmri, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    return DATA_DIR


if __name__ == '__main__':
    get_second_level_dataset()
