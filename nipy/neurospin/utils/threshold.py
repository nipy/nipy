"""
This module contains the function 
that thresholds an image and retains only the clusters of size>smin

Author : Bertrand Thirion, 2009
"""

#autoindent
import numpy as np
import scipy.stats as st
import nipy.neurospin.graph.field as ff
from nipy.io.imageformats import load, save, Nifti1Image 


# FIXME: 1) these functions should operate on ndarrays.
#        2) the test at the end of the code should be converted in a unit 
#           test (requires 1).


def threshold_scalar_image(iimage, oimage=None, th=0., smin=0, mask_image=None):
    """
    this function takes a 'grey level' threshold and a size threshold
    and gives as output an image where only the suprathreshold component
    of size > smin have not been thresholded out

    Parameters
    ----------
    iimage, string, path of a scalar input image
    oimage=None, string, path of the scalar output image
                 if None teh output image is not written
    th=0., float,  the chosen trheshold
    smin=0, int, cluster size threshold
    mask_image=None: a mask image to determine where in image this applies
                     if mask_image==None, the function is implied on 
                     where(image)
    
    returns
    -------
    output, image: the output image object

    Note, the 0 values of iimage are not considered so far
    """
    # fixme : add a header check here
    # 1. read the input image
    inim = load(iimage)
    ref_dim = inim.get_shape()
    x = inim.get_data()
    if mask_image==None:
        nvox = np.size(np.nonzero(x))
        xyz = np.array(np.where(x)).T.astype(np.int)
        m = (x!=0)
    else:
        mask = load(mask_image)
        m = mask.get_data()

    nvox = np.sum(m>0)
    x = x[np.where(m)]
    xyz = np.array(np.where(m)).T.astype(np.int)

    #2. threshold the map
    thx = np.zeros(nvox)
    supvox = np.sum(x>th)
    if supvox>0:
        # 2.a make a field based onthe thresholded image
        thim = ff.Field(supvox)
        thim.from_3d_grid(xyz[x>th])

        # 2.b get the cc and their size
        u = thim.cc()
        nsu = np.array([np.sum([u==i]) for i in range(u.max()+1)])
        invalid = nsu<smin
        u[invalid[u]]=-1
        
        # 2.c reorder u and write the final label map
        thx[x>th] = x[x>th]*(u>-1) 

    result = np.zeros(ref_dim)
    result[m>0] = thx
    onim = Nifti1Image(result.T,inim.get_affine())	
    onim.get_header()['descrip']= "thresholded image, threshold= %f,\ 
                                  cluster size=%d"%(th,smin)
    if oimage !=None:
       save(onim, oimage)	
    return onim


def threshold_z_image(iimage, oimage=None, corr=None, pval=None, smin=0, 
            mask_image=None, method=None):
    """
    this function takes a presumably gaussian image threshold and a
    size threshold and gives as output an image where only the
    suprathreshold component of size > smin have not been thresholded
    out This corresponds to a one-sided classical test the null
    hypothesis can be take to be the standard normal or the empiricall
    null.
    
    Parameters
    ----------
    iimage, string, the path of a presumably z-variate input image
    oimage=None, string, the path of the output image
    corr=None, string  the correction for multiple comparison method
               corr can be either None or 'bon' (Bonferroni) or 'fdr'
    pval=none, float, the desired classical p-value.
               the default behaviour of pval depends on corr
               if corr==None, pval = 0.001, else pval = 0.05
    smin=0, int, the  cluster size threshold
    mask_image=None, string path of a mask image to determine 
                     where thresholding is  applies
                     if mask_image==None, the function is implied 
                     on where(image)
    method=None: model of the null distribution:
                 if method==None: standard null
                 if method=='emp': empirical null
    
    Returns
    -------
    oimage: the output image
    """
    #?# 1.  read the image(s)
    nim = load(iimage)
    x = nim.get_data()
    if mask_image==None:
        nvox = np.size(np.nonzero(x)) 
        x = x[np.where(x)]
    else:
        mask = load(mask_image)
        m = mask.get_data()
        nvox = np.sum(m>0)
        x = x[np.where(m)]
        
    # 2. determine the threshold in z-variate
    #2.a determine the used-pval
    if pval==None:
        pval=0.001
        if corr=='bon':pval=0.05
        if corr=='fdr':pval=0.05

    # requires nvox !
    if corr=='bon':
        pval/=nvox
    th = st.norm.isf(pval)
    
    # requires the  data
    if corr=='fdr':
        if method=='emp':
            from emp_null import ENN
            efdr = ENN(x)
            th = efdr.threshold(pval,verbose=1)
        else:
            from emp_null import FDR
            lf = FDR(x)
            th = lf.threshold(pval)
            print th

    # 3. threshold the image 
    output_image = threshold_scalar_image(iimage, oimage, th=th, smin=smin,
                        mask_image=mask_image)
    return output_image




def _test_():

    from  os.path import join

    # Get the data
    nbru = range(1,13)
    nbeta = [29]
    theta = float(st.t.isf(0.01,100))
    smin = 25
    swd = "/tmp/"
    
    # a mask of the brain in each subject
    Mask_Images =["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/mask.img" % bru for bru in nbru]
    #Mask_Images =["/data/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/mask.img" % bru for bru in nbru]
    
    # activation image in each subject
    betas = [["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/spmT_%04d.img" % (bru, n) for n in nbeta] for bru in nbru]
    #betas = [["/data/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/spmT_%04d.img" % (bru, n) for n in nbeta] for bru in nbru]


    oimage = join(swd,'toto1.nii')
    threshold_scalar_image(betas[0][0],oimage,th=theta,smin=smin)
    oimage = join(swd,'toto2.nii')
    threshold_scalar_image(betas[0][0],oimage,th=theta,smin=smin,mask_image=Mask_Images[0])
    oimage = join(swd,'toto3.nii')
    threshold_z_image(betas[0][0],oimage,corr=None,smin=5)
    oimage = join(swd,'toto4.nii')
    threshold_z_image(betas[0][0],oimage,corr='fdr',smin=5)
    oimage = join(swd,'toto4.nii')
    threshold_z_image(betas[0][0],oimage,corr='fdr',smin=5,pval=0.2)
    oimage = join(swd,'toto5.nii')
    threshold_z_image(betas[0][0],oimage,corr='fdr',pval=0.1,smin=5,method='emp')
    oimage = join(swd,'toto6.nii')
    threshold_z_image(betas[0][0],oimage,corr='bon',smin=5)


