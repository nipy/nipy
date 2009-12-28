""" Short standalone script to deal with SPM's *_sn.mat and y_*nii transformations files. Use PyNifti and Scipy. """
import nifti
from numpy import *
import scipy.ndimage
from scipy.ndimage import affine_transform
#from numpy import *
from numpy import dot, isnan, any, allclose, diag, vstack, ones, nan_to_num, size
from numpy import zeros, indices, diag

# updated() is to update() what sorted() is to sort() - too bad it is not a dict built-in AFAIK
updated = lambda dic, newdic : dict(dic.items() + newdic.items())

def snmatfile_to_snparams(snfilename, target_nii = None):
	from scipy.io import loadmat
	sn = loadmat(snfilename)
	if target_nii and (target_nii.extent != tuple(sn["VG"][0][0].dim[0])):
		print "Warning : %s has different shape than originally used for the _sn.mat %s" % (target_nii.filename, sn["VG"][0][0].fname[0])
	return dot(sn["VF"][0][0].mat, sn["Affine"]), sn["Tr"]

def make_SPMdeform_from_SPMsnparams(target_nii, snparams):
	Mult, snTr = snparams
	def cosine_matrix(N, numreg):
		return matrix([sqrt((2. if k else 1.)/N) * cos(pi*(1+2*arange(N))*k/(2*N)) for k in range(numreg)]).T
	basX, basY, basZ = [ cosine_matrix(n, k) for n, k in zip(target_nii.extent, shape(snTr)) ]

	Def = [ zeros(target_nii.extent[:3], float32) for i in range(3) ]
	X, Y = indices(target_nii.extent[0:2]) + 1 # +1 because matlab counting

	for j in range(target_nii.extent[2]):
		tx, ty, tz = [ snTr[...,i] * basZ[j].T for i in range(3) ]
		X1 =   X + basX * tx * basY.T
		Y1 =   Y + basX * ty * basY.T
		Z1 = 1+j + basX * tz * basY.T
		Def[0][...,j] = Mult[0,0]*X1 + Mult[0,1]*Y1 + Mult[0,2]*Z1 + Mult[0,3]
		Def[1][...,j] = Mult[1,0]*X1 + Mult[1,1]*Y1 + Mult[1,2]*Z1 + Mult[1,3]
		Def[2][...,j] = Mult[2,0]*X1 + Mult[2,1]*Y1 + Mult[2,2]*Z1 + Mult[2,3]
	
	return nifti.NiftiImage(vstack([d.T[newaxis,newaxis,...] for d in Def]), updated(target_nii.header,{"scl_slope":1.}))

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
	if ( y_nii.extent[:3] == target_nii.extent[:3] and allclose(y_nii.sform, target_nii.sform) ):
		xyz_data = y_nii.data.reshape(3, -1)
	else: # need to resample the sampling-vector-field to accomodate the different output size
		ys = (nifti.NiftiImage(y, y_nii.header) for y in y_nii.data[:,0,...])
		yrs = [resample_NiftiImage(y, target_nii, order=1, mode='nearest') for y in ys]
		xyz_data = [i.data.ravel() for i in yrs]
	yvox = dot(src_nii.sform_inv, vstack((xyz_data, ones_like(xyz_data[0]))) )[:3]
	out = scipy.ndimage.map_coordinates(nan_to_num(src_nii.data.T), yvox, order=order)
	return nifti.NiftiImage(out.reshape(target_nii.data.shape), updated(target_nii.header,{"scl_slope":1.}))

if __name__ == '__main__':
	import sys
	try:
		srcfile, targetfile, deformfile, outputfile = sys.argv[1:5]
	except:
		sys.exit("""
Apply a SPM deformation field (y_*nii) or a SPM _sn parameter file (_sn*.mat)

Usage: %s srcfile.nii targetfile.nii [parameter_sn.mat | y_deform.nii | identity] outputfile.nii [order, y_output.nii ]
   Will output "outputfile.nii" which is the image of srcfile.nii through the
   deformation file (either parameter_sn.mat or y_deform.nii), with output
   dimensions of targetfile.nii. Optionally, the applied transformation is
   saved to y_output.nii. The order argument is the resampling order.

E.g: ROI_MNI_V4.nii t1_image.nii t1_image_seg_inv_sn.mat ROI_t1_space.nii.gz 0
   would resample the ROI_MNI label image to the t1 original nativespace using
   the _inv_sn.mat transformation file, with a nearest neighbour resampling to
   avoid interpolating labels.
""" % __file__)
	src_nii, target_nii = nifti.NiftiImage(srcfile), nifti.NiftiImage(targetfile)
	order = int(sys.argv[5]) if (len(sys.argv) >= 6) else 3
	if deformfile == 'identity':
		sys.exit(resample_NiftiImage(src_nii, target_nii, order = order).save(outputfile))
	if deformfile.endswith('.mat'):
		vox2mmw, Tr = snmatfile_to_snparams(deformfile, target_nii)
		nii_deforms = make_SPMdeform_from_SPMsnparams(target_nii, (vox2mmw, Tr))
	else:
		nii_deforms = nifti.NiftiImage(deformfile)
	if len(sys.argv) == 7:
		nii_deforms.save(sys.argv[6])
	apply_forward_SPM_deform(nii_deforms, src_nii, target_nii, order = order).save(outputfile)