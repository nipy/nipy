"""Short routines to apply SPM(tm) y_*.nii deformation fields"""
import scipy.ndimage
import nifti
from scipy.ndimage import affine_transform
#from numpy import *
from numpy import dot, isnan, any, allclose, diag, vstack, ones, nan_to_num, size

def resample_NiftiImage(src, target, order = 3, mode = 'constant'):
    affine = dot(src.sform_inv, target.sform)
    af, of = affine[:3, :3], affine[:3, -1]
    srcdata = src.data if not any(isnan(src.data)) else nan_to_num(src.data)
    if allclose(af, diag(diag(af))): # if scaling matrix
        af = af.diagonal()
        of /= af
    a = affine_transform(srcdata.T, af, of, output_shape=target.data.T.shape, order = order, mode = mode).T
    return nifti.NiftiImage(a, target.header)

def apply_forward_SPM_deform(y_nii, src_nii, target_nii, order = 2):
    """ Apply an SPM-like deformation field (y_*.nii images) to the src_nii image.
 Output dimensions and coordinates are copied from target_nii.
 order : spline order for interpolation (1 is fastest but less accurate)"""
    ys = (nifti.NiftiImage(y, y_nii.header) for y in y_nii.data[:,0,...])
    yrs = [resample_NiftiImage(y, target_nii, order=1, mode='nearest') for y in ys]
    yvox = dot(src_nii.sform_inv, vstack([i.data.ravel() for i in yrs]+[ones(size(i.data))]))[:3]
    out = scipy.ndimage.map_coordinates(nan_to_num(src_nii.data.T), yvox, order=order)
    return nifti.NiftiImage(out.reshape(target_nii.data.shape), target_nii.header)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print ("Apply a SPM-software deformation (usually named as y_*.nii) to an image")
        print("Args : deform_field_file source_image_file target_image_file output_filename")
        print ("eg. python %s y_mprage000007705688.nii.gz c1mprage000007705688.nii.gz wmmprage000007705688.nii.gz output_wc1.nii" % sys.argv[0])
    else:
        if not (sys.argv[4].endswith('.nii') or sys.argv[4].endswith('.nii.gz')):
            print("Last argument _must_ end with .nii or .nii.gz")
        else:
            imgs = [nifti.NiftiImage(x) for x in sys.argv[1:4]]
            output = apply_forward_SPM_deform(*imgs)
            output.save(sys.argv[4])
