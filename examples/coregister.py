"""Example using Tom's registration code from scipy.

"""

from os import path
from glob import glob

import numpy as np

import scipy.ndimage._registration as reg

# Data files
basedir = '/Users/cburns/data/twaite'
anatfile = path.join(basedir, 'ANAT1_V0001.img')
funcdir = path.join(basedir, 'fMRIData')
fileglob = path.join(funcdir, 'FUNC1_V000?.img')   # Get first 10 images


def demo_coregistration(anatfile, funclist, optimizer_method='powell', 
                        histo_method=1, smooth_histo=0, smooth_image=0, 
                        ftype=1):
    """Coregister functional images to the anatomical image.

    Each fMRI volume is first perturbed (rotated, translated) by a random 
    value. The initial registration is measured, then the optimal alignment is 
    computed and the registration measure made following the volume remap.
    The fMRI registration is done with the first fMRI volume using normalized 
    cross-correlation.  Each fMRI volume is rotated to the fMRI-0 volume and 
    the series is ensemble averaged.  The ensemble averaged is then registered 
    with the anatomical MRI volume using normalized mutual information.
    The fMRI series is then rotated with this parameter. The alignments are 
    done with 3D cubic splines.

    Parameters
    ----------
    anatfile : filename
        path to the anatomical file
    funclist : list
        list of paths to functional images
    

    Returns
    -------
    measures : ndarray
    imageF_anat : dict
    fmri_series : dict

    Notes
    -----
    show results with

    In [59]: measures[25]['cost']
    Out[59]: -0.48607185

    In [60]: measures[25]['align_cost']
    Out[60]: -0.99514639

    In [61]: measures[25]['align_rotate']
    Out[61]:
    array([ 1.94480181,  5.64703989,  5.35002136, -5.00544405, -2.2712214, -1.42249691], dtype=float32)

    In [62]: measures[25]['rotate']
    Out[62]:
    array([ 1.36566341,  4.70644331,  4.68198586, -4.32256889, -2.47607017, -2.39173937], dtype=float32)


    """

    # read the anatomical MRI volume
    anat_desc = reg.load_anatMRI_desc()
    imageF_anat = reg.load_volume(anat_desc, imagename=anatfile)
    # the sampling structure
    imdata = reg.build_structs()
    # the volume filter
    imageF_anat['fwhm'] = reg.build_fwhm(imageF_anat['mat'], imdata['step'])

    # read in the file list of the fMRI data
    metric_test = np.dtype([('cost', 'f'),
                           ('align_cost', 'f'),
                           ('rotate', 'f', 6),
                           ('align_rotate', 'f', 6)])

    #fMRIdata = read_fMRI_directory('fMRIData\*.img')
    #fMRIdata = read_fMRI_directory(funcdir + '/*.img')
    fMRIdata = funclist
    fmri_desc = reg.load_fMRI_desc()
    fmri_series = {}
    ave_fMRI_volume = np.zeros(fmri_desc['layers']*fmri_desc['rows']
                               *fmri_desc['cols'],
                      dtype=np.float64).reshape(fmri_desc['layers'], 
                                                fmri_desc['rows'], 
                                                fmri_desc['cols'])
    count = 0
    number_volumes = len(fMRIdata)
    measures = np.zeros(number_volumes, dtype=metric_test)
    # load and perturb (rotation, translation) the fMRI volumes
    for i in fMRIdata:
        image = reg.load_volume(fmri_desc, i)
        # random perturbation of angle, translation for each volume beyond 
        # the first
        if count == 0:
            image['fwhm'] = reg.build_fwhm(image['mat'], imdata['step'])
            fmri_series[count] = image
            count = count + 1
        else:
            x = np.random.random(6) - 0.5
            x = 10.0 * x
            fmri_series[count] = reg.demo_rotate_fMRI_volume(image, x)
            measures[count]['rotate'][0:6] = x[0:6]
            count = count + 1

    # load and register the fMRI volumes with volume_0 using normalized 
    # cross correlation metric
    imageF = fmri_series[0]
    if smooth_image:
        image_F_xyz = reg.filter_image_3D(imageF['data'], imageF['fwhm'], ftype)
        imageF['data'] = image_F_xyz
    for i in range(1, number_volumes):
        imageG = fmri_series[i]
        # the measure prior to alignment 
        measures[i]['cost'] = reg.check_alignment(imageF, imageG, imdata, 
                                                  method='ncc',
                                                  lite=histo_method, 
                                                  smhist=smooth_histo)
        x = reg.python_coreg(imageF, imageG, imdata, lite=histo_method, 
                         method='ncc', opt_method=optimizer_method, 
                         smhist=smooth_histo, smimage=smooth_image)
        measures[i]['align_rotate'][0:6] = x[0:6]
        measures[i]['align_cost'] = reg.check_alignment(imageF, imageG, imdata,
                                                        method='ncc', 
                                                        lite=histo_method, 
                                                        smhist=smooth_histo,
                                                        alpha=x[0], 
                                                        beta=x[1], 
                                                        gamma=x[2], 
                                                        Tx=x[3], 
                                                        Ty=x[4], 
                                                        Tz=x[5])


    # align the volumes and average them for co-registration with the 
    # anatomical MRI 
    ave_fMRI_volume = fmri_series[0]['data'].astype(np.float64)
    for i in range(1, number_volumes):
        image = fmri_series[i]
        x[0:6] = measures[i]['align_rotate'][0:6]
        # overwrite the fMRI volume with the aligned volume
        fmri_series[i] = reg.remap_image(image, x, resample='cubic')
        ave_fMRI_volume = ave_fMRI_volume + fmri_series[i]['data'].astype(np.float64)

    ave_fMRI_volume = (ave_fMRI_volume / float(number_volumes)).astype(np.uint8)
    ave_fMRI_volume = {'data' : ave_fMRI_volume, 'mat' : imageF['mat'], 
                       'dim' : imageF['dim'], 'fwhm' : imageF['fwhm']}
    # register (using normalized mutual information) with the anatomical MRI
    if smooth_image:
        image_F_anat_xyz = reg.filter_image_3D(imageF_anat['data'], 
                                           imageF_anat['fwhm'], ftype)
        imageF_anat['data'] = image_F_anat_xyz
    x = reg.python_coreg(imageF_anat, ave_fMRI_volume, imdata, 
                         lite=histo_method, method='nmi', 
                         opt_method=optimizer_method, smhist=smooth_histo, 
                         smimage=smooth_image)
    print 'functional-anatomical align parameters '
    print x
    for i in range(number_volumes):
        image = fmri_series[i]
        # overwrite the fMRI volume with the anatomical-aligned volume
        fmri_series[i] = reg.remap_image(image, x, resample='cubic')

    return measures, imageF_anat, fmri_series


if __name__ == '__main__':
    print 'Coregister anatomical:\n', anatfile
    print '\nWith these functional images:'
    funclist = glob(fileglob)
    funclist = funclist[0:4]
    for func in funclist:
        print func
    #measures, imageF_anat, fmri_series = \
    #reg.demo_MRI_coregistration(anatfile, funclist[0:4])
    demo_coregistration(anatfile, funclist)
