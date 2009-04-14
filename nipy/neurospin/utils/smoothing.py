"""
Routine for smoothing data using diffusion on graphs.
Works well only for small kernels.

Mybe should be abandoned

Author : Bertrand Thirion, 2006-2009
"""

import numpy as np
import nipy.neurospin.graph.field as ff

def cartesian_smoothing(ijk,data,sigma):
	"""
	Smoothing data on a(n uncomplete) cartesian grid
	INPUT:
	-ijk : list of the positions, which is assumed to be an (n,3) int array
	it is typically returned by transpose(numpy.where())
	- data : an array of data sampled from the grid size(ijk.shape[0],d)
	where d is the datas dimension
	-sigma : the kernel parameter
	OUTPUT
	- data, which is the smoothed data
	"""
	if ijk.shape[1]!=3:
		raise ValueError, "please provide a (n,3) position array"
	ijk = ijk.astype('i')
	n = ijk.shape[0]
	if data.shape[0]!=n:
		raise ValueError, "incompatible dimension for data and ijk"
	data = data.astype('f')
	if np.size(data)==n:
		data = np.reshape(data,(n,1))
	
	f = ff.Field(n)
	f.from_3d_grid(ijk,6)
	f.normalize(1)
	D = f.weights
	L = D*6-6*(D==0)
	f.set_weights(L)

	niter = max(10,int(1+0.5*sigma*sigma*(1+np.absolute(L).max())))
	
	kp = 0.5*sigma*sigma/niter
	for i in range(niter):
		#print data
		f.set_field(data.copy())
		f.diffusion()
		data = data + kp * f.field
				
	return data


def check_smoothing():

	import nifti
	offset = 1.0
	# generate some data
	x = offset*np.ones((30,30,30),'f')
	ijk = np.where(np.ones((30,30,30)))
	ref = np.array([15,15,15])
	x[ref[0],ref[1],ref[2]]+=1;
	sigma = 1.87

	# take the ideal result
	dx = np.sum((np.transpose(ijk)-ref)*(np.transpose(ijk)-ref),1)
	target = 1/(2*np.pi*sigma**2)**(1.5) * np.exp(-dx/(2*sigma**2))
	target = target/np.sum(target)
	target += offset
	gaussian = x.copy()
	gaussian[ijk] = target
	nifti.NiftiImage(gaussian).save("/tmp/target.nii")

	# compute the result with the cartesian_smoothing procedure
	data = np.reshape(x.copy(),(np.size(x),1))
	trial = x.copy()
	trial[ijk] = np.squeeze(cartesian_smoothing(np.transpose(ijk),data,sigma))
	nifti.NiftiImage(trial).save("/tmp/result.nii")
	
	dt = gaussian-trial
	print "error:", (dt*dt).sum()
	return gaussian, trial

