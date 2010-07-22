# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Get two images from the web (one mask image and one spmT image) and
put them in the dir: ~/.nipy/tests/data
should be quick and light (<6MB)

Author : Bertrand Thirion, 2009
"""

import os
import urllib2
import tarfile

def get_it():
    """
    light dataset for multi-subject analysis
    """
    # define several paths
    url = 'ftp://ftp.cea.fr/pub/dsv/madic/download/nipy'
    data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
    MaskImage = os.path.join(data_dir,'mask.nii.gz')
    InputImage = os.path.join(data_dir,'spmT_0029.nii.gz')
    GroupData = os.path.join(data_dir,'group_t_images.tar.gz')

    # possibly create data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        assert os.path.exists(data_dir)

    # download MaskImage if necessary
    if os.path.exists(MaskImage)==False:
        filename = 'mask.nii.gz'
        datafile = os.path.join(url,filename)
        fp = urllib2.urlopen(datafile)
        local_file = open(MaskImage, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    # download InputImage if necessary
    if os.path.exists(InputImage)==False:
        filename = 'spmT_0029.nii.gz'
        datafile = os.path.join(url,filename)
        fp = urllib2.urlopen(datafile)
        local_file = open(InputImage, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    # download GroupData if necessary
    if os.path.exists(GroupData)==False:
        filename = 'group_t_images.tar.gz'
        datafile = os.path.join(url,filename)
        fp = urllib2.urlopen(datafile)
        local_file = open(GroupData, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    #untargzip GroupData
    tar = tarfile.open(GroupData)
    tar.extractall(data_dir)
    tar.close()
    os.remove(GroupData)

def get_localizer_dataset():
    """
    Heavier dataset (30 MO) for first-level analysis
    """
    # define several paths
    url = 'ftp://ftp.cea.fr/pub/dsv/madic/download/nipy'
    data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
    raw_fmri = os.path.join(data_dir,'s12069_swaloc1_corr.nii.gz')
    paradigm = os.path.join(data_dir,'localizer_paradigm.csv')

    # create data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        assert os.path.exists(data_dir)

    # download MaskImage if necessary
    if os.path.exists(paradigm)==False:
        print 'Downloading mask image, this make take time'
        datafile = os.path.join(url, 'localizer_paradigm.csv')
        fp = urllib2.urlopen(datafile)
        local_file = open(paradigm, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    # download raw_fmri if necessary
    if os.path.exists(raw_fmri)==False:
        print 'Downloading fmri image, this make take time'
        filename = 's12069_swaloc1_corr.nii.gz'
        datafile = os.path.join(url,filename)
        fp = urllib2.urlopen(datafile)
        local_file = open(raw_fmri, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

if __name__ == '__main__':
    get_it()
