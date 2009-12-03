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

# rmk: These functions could use scipy.ndimage.label and thus not rely
# on any nipy C code.

def threshold_array(data_array, mask_array=None, th=0., smin=0, nn=18):
    """
    Array thresholding facility    

    Parameters
    ----------
    data_array, array 
                the data to threshold in its positions
    mask_array=None, array
                     the positsion to consider within data_array
                     if None, all teh original image is considered
    th=0., float , teh scalar threshold to apply
    smin=0, int, cluster size threshold
    nn=18, int spatial neighboring system: 6,18 or 26

    Returns
    -------
    thx array of shape (sum(mask)) that yields the values that remain
    """
    if mask_array==None:
        mask_array = np.ones(np.shape(data_array))
    if np.shape(data_array)!=np.shape(mask_array):
        raise ValueError, "mask_array and data_array do not have the same shape"
    data_array = data_array[mask_array>0]
    xyz = np.array(np.where(mask_array>0)).T.astype(np.int)

    #2. threshold the map
    nvox = np.sum(mask_array>0)
    thx = np.zeros(nvox)
    supvox = np.sum(data_array>th)
    if supvox>0:
        # 2.a make a field based onthe thresholded image
        thim = ff.Field(supvox)
        thim.from_3d_grid(xyz[data_array>th])

        # 2.b get the cc and their size
        u = thim.cc()
        nsu = np.array([np.sum([u==i]) for i in range(u.max()+1)])
        invalid = nsu<smin
        u[invalid[u]]=-1
        
        # 2.c reorder u and write the final label map
        thx[data_array>th] = data_array[data_array>th]*(u>-1) 
    return thx


def threshold_scalar_image(iimage, oimage=None, th=0., smin=0, nn=18, 
        mask_image=None):
    """
    this function takes a 'grey level' threshold and a size threshold
    and gives as output an image where only the suprathreshold component
    of size > smin have not been thresholded out

    Parameters
    ----------
    iimage, string, path of a scalar input image
    oimage=None, string, path of the scalar output image
        if None the output image is not written
    th=0., float,  the chosen trheshold
    smin=0, int, cluster size threshold
    nn=18, int spatial neighboring system: 6,18 or 26
    mask_image=None: a mask image to determine where in image this applies
                     if mask_image==None, the function is implied on 
                     where(image)
    
    Returns
    -------
    output, image: the output image object

    Note, the 0 values of iimage are not considered so far
    """
    # FIXME: add a header check here
    # 1. read the input image
    
    if mask_image==None:
        m = None
    else:
        mask = load(mask_image)
        m = mask.get_data()

    inim = load(iimage)    
    x = inim.get_data()
    
    thx = threshold_array(x, m, th, smin, nn=nn)
    
    ref_dim = inim.get_shape()
    result = np.zeros(ref_dim)
    result[m>0] = thx
    onim = Nifti1Image(result.T,inim.get_affine())	
    onim.get_header()['descrip']= "thresholded image, threshold= %f,\
                                   cluster size=%d"%(th,smin)
    if oimage !=None:
       save(onim, oimage)	
    return onim


def threshold_z_array(data_array, mask_array=None, correction=None, 
                                  pval=None, smin=0, nn=18, method=None):
    """
    threshold an array at a secified p-value, with/out correction 
    for multiple compaison
    
    Parameters
    ----------
    data_array, array 
                the data to threshold in its positions
    mask_array: array or None, optional
        the position to consider within data_array if None, all the 
        original image is considered
    correction: {None, 'bon', 'fdr'}
        the correction for multiple comparison method correction can be 
        either None or 'bon' (Bonferroni) or 'fdr'
    pval: float or None, optional
        The desired classical p-value. The default behaviour of pval 
        depends on correction if correction==None, pval = 0.001, 
        else pval = 0.05
    smin: int, optional
        The cluster size threshold
    method: 'emp' or None 
        model of the null distribution:
        if method==None: standard null
        if method=='emp': empirical null
    """
    if mask_array==None:
        mask_array = np.ones(np.shape(data_array))
    if np.shape(data_array)!=np.shape(mask_array):
        raise ValueError, "mask_array and data_array do not have\
                          the same shape"
    nvox = np.sum(mask_array)    

    # 1. determine the threshold in z-variate
    #1.a determine the used-pval
    if pval==None:
        pval=0.001
        if correction=='bon':pval=0.05
        if correction=='fdr':pval=0.05

    # requires nvox !
    if correction=='bon':
        pval/=nvox
    th = st.norm.isf(pval)
    
    # requires the  data
    if correction=='fdr':
        if method=='emp':
            from emp_null import ENN
            efdr = ENN(data_array[mask_array>0])
            th = efdr.threshold(pval,verbose=0)
        else:
            from emp_null import FDR
            lf = FDR(data_array[mask_array>0])
            th = lf.threshold(pval)
    print 'the threhsold is:', th

    # 3. threshold the image 
    return threshold_array(data_array, mask_array, th, smin, nn)


def threshold_z_image(iimage, oimage=None, correction=None, pval=None, smin=0, 
            nn=18, mask_image=None, method=None):
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
    correction=None, string  the correction for multiple comparison method
               correction can be either None or 'bon' (Bonferroni) or 'fdr'
    pval=none, float, the desired classical p-value.
               the default behaviour of pval depends on correction
               if correction==None, pval = 0.001, else pval = 0.05
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
    if mask_image==None:
        m = None
    else:
        mask = load(mask_image)
        m = mask.get_data()
    nim = load(iimage)
    x = nim.get_data()
        
    thx = threshold_z_array(x, m, correction, pval, smin, nn, method)
    
    ref_dim = nim.get_shape()
    result = np.zeros(ref_dim)
    result[m>0] = thx
    onim = Nifti1Image(result.T, nim.get_affine())	
    onim.get_header()['descrip']= "thresholded image, threshold= %f,\
                                   cluster size=%d"%(thx, smin)
    if oimage !=None:
       save(onim, oimage)	
    return onim


