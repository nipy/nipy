
import numpy as np
import math
import glob
import flat_file_IO as ffi
import neuroimaging.fixes.scipy.ndimage._registration as reg
import neuroimaging.fixes.scipy.ndimage._register as r
import scipy.ndimage.interpolation as ndi
from neuroimaging.core.image import image


def volume_remap(a_f_affine, f_f_affine, fmri_series):

    """
    takes the raw fmri_series, the functional-anatomical mapping
    and the dictionary (array) of functional-functional mappings
    and builds the composite for each fMRI volume. Then it remaps
    to the space of functional[0] and anatomical. the original fMRI
    series and the remapped series get converted to 4D nd_arrays

    Parameters 
    ----------

    a_f_affine : {nd_array}
        the affine (4x4) matrix that registers the ensemble-averaged fMRI volume
	with the anatomical volume.

    f_f_affine : {dict}
        dictionary that contains 4x4 (nd_array) affine matrices that map volume i
	with volume 0

    fmri_series : {list}
        each element of the list is the raw fMRI 3D volume


    Returns 
    -------

    fmri_series : {nd_array}
        4D array of the raw fMRI volumes
    
    new_fmri_series : {nd_array}
        4D array of the remapped fMRI volumes



to run:
fmri_series, new_fmri_series = nreg.volume_remap(f_a_affine, f_f_affine, fmri_series)

    """

    if type(a_f_affine) != np.ndarray:
	raise ValueError, "anatomical-functional matrix is not ndarray"

    if a_f_affine.shape != (4,4):
        raise ValueError, "anatomical-functional matrix is not 4x4"

    if type(f_f_affine) != dict:
	raise ValueError, "functional-functional array must be a dictionary"

    if len(f_f_affine) <= 1:
	raise ValueError, "must have more than one f_f_affine array"

    if f_f_affine[1].shape != (4,4):
        raise ValueError, "functional-functional matrix is not 4x4"

    if type(fmri_series) != list:
	raise ValueError, "fmri_series must be a list"

    if len(fmri_series) <= 1:
	raise ValueError, "must have more than one functional volume in the list array"

    if fmri_series[0].ndim != 3:
	raise ValueError, "functional volumes must be 3D"


    new_fmri_series = []
    number_volumes  = len(fmri_series)


    image = fmri_series[0]
    M     = a_f_affine

    new_fmri_series.append(reg.remap_image(image, M, resample='cubic'))

    for i in range(1, number_volumes):
        image = fmri_series[i]
        # 
	# build composite (f_f * f_a) affine
        # 
        M = np.dot(a_f_affine, f_f_affine[i])
        # 
        # add the new aligned volume
        # 
        new_fmri_series.append(reg.remap_image(image, M, resample='cubic'))


    # convert the raw and aligned series into 4D numpy arrays 
    fmri_series     = np.asarray(fmri_series)
    new_fmri_series = np.asarray(new_fmri_series)

    return fmri_series, new_fmri_series


def coregistration(fMRIdata, anatfile=None, optimizer_method='hybrid', histo_lite=1, 
                   smooth_histo=0, smooth_image=0, ftype=1):

    """
    inputs an anatomical MRI and list of fMRI's and registers as follows:
    [1] register fMRI[i] to fMRI[0]
    [2] build the ensemble average of the aligned fMRI
    [3] register the ensemble average to the anatomical
    return the measures (original correlation, aligned correlation -- both as negative costs--
    and the 6 parameters (3 rotation, 3 translations) required for registration and the
    original fMRI series for later permanent remap, the func-anat affine and the func-func 
    affine array


    Parameters 
    ----------

    fMRIData : {list}
        the list of names of the fMRI volumes.

    anatfile : {str}, optional
        the name of the anatomical MRI. if None, then do not align functional to anatomical
	and create f_a_affine as eye(4)

    optimizer_method : {'powell', 'cg', 'hybrid'}, optional
        registration is two pass. Pass 1 is low res to get close to alignment
        and pass 2 starts at the pass 1 optimal alignment. In powell pass 1 and
        2 are powell, in hybrid pass 2 is conjugate gradient.

    histo_lite : {0, 1}, optional
        histo_lite of 1 is to jitter both images during resampling. 0
        is to not jitter. jittering is for non-aliased volumes.

    smooth_histo: {0, 1}, optional
        flag for joint histogram low pass filtering. 0 for no filter,
        1 for do filter.

    smooth_image: {0, 1}, optional
        flag for 3D filtering of the volumes prior to registration. 

    ftype: {1, 2}, optional
        select the type of filter kernel. type 1 is the SPM Gaussian
	convolved with b-spline. type 2 is Gaussian.


    Returns 
    -------

    measures : {nd_array}
        of type metric_test. contains the (negative) correlation of
	volume i and volume 0 before (['cost']) and after ([align_cost'])
	registration. also stores the 6 parameters (3 angle, 3 translations)
	that form the affine matrix that registers volume i with volume 0. these
	are the negative of the parameters that register volume 0 with volume i

    M_f_a : {nd_array}
        the affine (4x4) matrix that registers the ensemble-averaged fMRI volume
	with the anatomical volume.

    f_f_affine : {dict}
        dictionary that contains 4x4 (nd_array) affine matrices that map volume i
	with volume 0

    fmri_series : {list}
        each element of the list is the raw fMRI 3D volume
        

to run:
fMRIData = ffi.read_fMRI_directory('fMRI_Random\*.img')
measures, f_a_affine, f_f_affine, fmri_series = nreg.coregistration(fMRIData, 'ANAT1_V0001.img')
measures, f_a_affine, f_f_affine, fmri_series = nreg.coregistration(fMRIData')

    """

    if anatfile !=None and type(anatfile) != str:
	raise ValueError, "anatomical must be a string"

    if type(fMRIdata) != list:
	raise ValueError, "fMRIdata must be a list"

    if type(fMRIdata[0]) != str:
	raise ValueError, "fMRIdata elements must be a string"

    if optimizer_method != 'powell' and optimizer_method != 'cg'  and optimizer_method != 'hybrid':
        raise ValueError, "only optimize methods powell, cg or hybrid are supported"

    if histo_lite != 0 and histo_lite != 1: 
        raise ValueError, "choose histogram generation type 0 or 1 only"

    if smooth_image != 0 and smooth_image != 1: 
        raise ValueError, "smooth_image must be 0 or 1 only"

    if ftype != 0 and ftype != 1: 
        raise ValueError, "choose filter type 0 or 1 only"

    anat_desc = ffi.load_anatMRI_desc()
    fmri_desc = ffi.load_fMRI_desc()

    if anatfile != None:
        # read the anatomical MRI volume, given the file path
        imageG_anat, anat_mat = ffi.load_MRI_volume(anat_desc, imagename=anatfile)
        # imageG_anat = image.load(anatfile)
        # imageG_anat = np.asarray(imageG_anat)
        # scale to 8 bits using the integrated histogram
        imageG_anat = reg.scale_image(imageG_anat)

    # allocate the structure for the processed fMRI array
    metric_test = np.dtype([('cost', 'f'),
                            ('align_cost', 'f'),
                            ('align_parameters', 'f', 6)])

    # read in the fmri_mat of the fMRI data. replaced with
    # image_fmri = image.load(fMRIdata[0])
    # fmri_mat   = image_fmri.affine
    # ...
    # NOTE: require that xdim is col-row 0, ydim is col-row 1, zdim is col-row 2
    # NOTE: the matrix will be diagonal. non-diagonal is a shear.
    # ...
    # image_fmri = np.asarray(image_fmri)
    image_fmri, fmri_mat = ffi.load_MRI_volume(fmri_desc, fMRIdata[0])

    # the sampling structure
    step = np.array([1, 1, 1], dtype=np.int32)

    # one time build of the fwhm that is used to build the filter kernels
    if anatfile != None:
        anat_fwhm = reg.build_fwhm(anat_mat, step)
    fmri_fwhm = reg.build_fwhm(fmri_mat, step)

    # blank volume that will be used for ensemble average for fMRI volumes
    # prior to functional-anatomical coregistration

    number_volumes = len(fMRIdata)
    measures = np.zeros(number_volumes, dtype=metric_test)

    # allocate the empty dictionary that will contain f_f affine 
    f_f_affine = {} 
    # allocate the empty list that will contain metrics and aligned volumes 
    fmri_series = []
    new_fmri_series = []

    for i in fMRIdata:
        image, junkmat = ffi.load_MRI_volume(fmri_desc, i)
        # image = image.load(i)
        # image = np.asarray(image)

	# the data is pre-scaled. just need to recast to uint8
        image = reg.scale_image(image)
        fmri_series.append(image)

    # load and register the fMRI volumes with volume_0 using normalized
    # cross correlation metric
    imageG = fmri_series[0]
    if smooth_image:
        imageG = reg.filter_image_3D(imageG, fmri_fwhm, ftype)
    for i in range(1, number_volumes):
        imageF = fmri_series[i]
        if smooth_image:
            imageF = reg.filter_image_3D(imageF, fmri_fwhm, ftype)
        # the measure prior to alignment 
        measures[i]['cost'] = reg.check_alignment(imageG, fmri_mat, imageF, fmri_mat,
                                                  method='ncc',lite=histo_lite,
					          smhist=smooth_histo)
        f_f = reg.register(imageG, fmri_mat, imageF, fmri_mat, lite=histo_lite,
                           method='ncc', opt_method=optimizer_method, smhist=smooth_histo)
        measures[i]['align_parameters'][0:6] = f_f[0:6]
	# I hard-wirred the fMRI intra-modal registration to be 'ncc' (cross correlation)
        measures[i]['align_cost'] = reg.check_alignment(imageG, fmri_mat, imageF, fmri_mat,
                                                        method='ncc', lite=histo_lite,
						        smhist=smooth_histo, alpha=f_f[0],
                                                        beta=f_f[1], gamma=f_f[2], Tx=f_f[3],
						        Ty=f_f[4], Tz=f_f[5])


    # align the volumes and average them for co-registration with the anatomical MRI 
    ave_fMRI_volume = fmri_series[0].astype(np.float64)
    new_fmri_series.append(fmri_series[0])
    for i in range(1, number_volumes):
        image = fmri_series[i]
        x = measures[i]['align_parameters'][0:6]
	M = reg.get_inverse_mappings(x)
        f_f_affine[i] = M
        if anatfile!=None:
            # overwrite the fMRI volume with the aligned volume
            new_fmri_series.append(reg.remap_image(image, M, resample='cubic'))
            ave_fMRI_volume = ave_fMRI_volume + new_fmri_series[i].astype(np.float64)


    if anatfile==None:
	M_f_a = np.eye(4)
        return measures, M_f_a, f_f_affine, fmri_series
    else:
        ave_fMRI_volume = (ave_fMRI_volume / float(number_volumes)).astype(np.uint8)

    # register (using normalized mutual information) with the anatomical MRI
    if smooth_image:
        imageG_anat = reg.filter_image_3D(imageG_anat, anat_fwhm, ftype)

    # I hard-wirred the MRI-fMRI intra-modal registration to be 'nmi' (norm mutual information)
    f_a = reg.register(imageG_anat, anat_mat, ave_fMRI_volume, fmri_mat, lite=histo_lite,
                       method='nmi', opt_method=optimizer_method, smhist=smooth_histo)

    M_f_a = reg.get_inverse_mappings(f_a)

    return measures, M_f_a, f_f_affine, fmri_series


def dual_registration(image1, affine1, image2, affine2, optimizer_method='hybrid', cost='ncc',
                      histo_lite=1, smooth_histo=0, smooth_image=0, ftype=1):


    """
    registration of two volumes. volumes can be intra- or inter-modal. used for
    testing and later registration for PET/MRI, etc. that would explore gradient
    cost functions.

    Parameters 
    ----------

    image1 : {nd_array} 
        image1 is the source image to be remapped during the registration. 

    affine1 : {nd_array} 
        affine1 is the source image MAT 

    image2 : {nd_array} 
        image2 is the reference image that image1 gets mapped to. 

    affine2 : {nd_array} 
        affine2 is the source image MAT 

    optimizer_method : {'powell', 'cg', 'hybrid'}, optional
        registration is two pass. Pass 1 is low res to get close to alignment
        and pass 2 starts at the pass 1 optimal alignment. In powell pass 1 and
        2 are powell, in hybrid pass 2 is conjugate gradient.

    cost: {'nmi', 'mi', 'ncc', 'ecc', 'mse'}, optional
        flag for type of registration metric. nmi is normalized mutual
        information; mi is mutual information; ecc is entropy cross
        correlation; ncc is normalized cross correlation. mse is mean
	squared error.

    histo_lite : {0, 1}, optional
        histo_lite of 1 is to jitter both images during resampling. 0
        is to not jitter. jittering is for non-aliased volumes.

    smooth_histo: {0, 1}, optional
        flag for joint histogram low pass filtering. 0 for no filter,
        1 for do filter.

    smooth_image: {0, 1}, optional
        flag for 3D filtering of the volumes prior to registration. 

    ftype: {1, 2}, optional
        select the type of filter kernel. type 1 is the SPM Gaussian
	convolved with b-spline. type 2 is Gaussian.

    Returns 
    -------

    measures : {nd_array}
        of type metric_test. contains the (negative) correlation of
	volume i and volume 0 before (['cost']) and after ([align_cost'])
	registration. also stores the 6 parameters (3 angle, 3 translations)
	that form the affine matrix that registers volume i with volume 0. these
	are the negative of the parameters that register volume 0 with volume i


    """

    # allocate the structure for the processed fMRI array
    metric_test = np.dtype([('cost', 'f'),
                            ('align_cost', 'f'),
                            ('align_parameters', 'f', 6)])

    measures = np.zeros(1, dtype=metric_test)

    image1 = reg.scale_image(image1)
    image2 = reg.scale_image(image2)

    # the sampling structure
    step = np.array([1, 1, 1], dtype=np.int32)

    image1_fwhm = reg.build_fwhm(affine1, step)
    image2_fwhm = reg.build_fwhm(affine2, step)

    if smooth_image:
        image1 = reg.filter_image_3D(image1, image1_fwhm, ftype)
        image2 = reg.filter_image_3D(image2, image2_fwhm, ftype)


    # the measure prior to alignment 
    measures['cost'] = reg.check_alignment(image1, affine1, image2, affine2,
                                           method='ncc',lite=histo_lite,
		                           smhist=smooth_histo)
    x = reg.register(image1, affine1, image2, affine2, lite=histo_lite, multires=[4, 2],
                     method=cost, opt_method=optimizer_method, smhist=smooth_histo)
    measures['align_parameters'][0:6] = x[0:6]
    measures['align_cost'] = reg.check_alignment(image1, affine1, image2, affine2,
                                                 method='ncc', lite=histo_lite,
			                         smhist=smooth_histo, alpha=x[0],
                                                 beta=x[1], gamma=x[2], Tx=x[3],
					         Ty=x[4], Tz=x[5])

    return measures


