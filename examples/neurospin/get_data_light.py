"""
Get two images from the web (one mask image and one spmT image) and
put them in the dir: ~/.nipy/tests/data
should be quick and light (<3MB)

Author : Bertrand Thirion, 2009
"""

import os
import urllib2
import tarfile

def getIt():
    """
    """
    # define several paths
    url = 'ftp://ftp.cea.fr/pub/dsv/madic/download/nipy'
    data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
    MaskImage = os.path.join(data_dir,'mask.nii.gz')
    InputImage = os.path.join(data_dir,'spmT_0029.nii.gz')
    GroupData = os.path.join(data_dir,'group_t_images.tar.gz')

    # download MaskImage if necessary
    if os.path.exists(MaskImage)==False:
        filename = 'mask.nii.gz'
        datafile = os.path.join(url,filename)
        fp = urllib2.urlopen(datafile)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            assert os.path.exists(data_dir)
        local_file = open(MaskImage, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    # download InputImage if necessary
    if os.path.exists(InputImage)==False:
        filename = 'spmT_0029.nii.gz'
        datafile = os.path.join(url,filename)
        fp = urllib2.urlopen(datafile)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            assert os.path.exists(data_dir)
        local_file = open(InputImage, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    # download GroupData if necessary
    if os.path.exists(GroupData)==False:
        filename = 'group_t_images.tar.gz'
        datafile = os.path.join(url,filename)
        fp = urllib2.urlopen(datafile)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            assert os.path.exists(data_dir)
        local_file = open(GroupData, 'w')
        local_file.write(fp.read())
        local_file.flush()
        local_file.close()

    #untargzip GroupData
    tar = tarfile.open(GroupData)
    tar.extractall(data_dir)
    tar.close()
    os.remove(GroupData)
    

if __name__ == '__main__':
    getIt()
