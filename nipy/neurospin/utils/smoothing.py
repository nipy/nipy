"""
Routine for smoothing data using diffusion on graphs.
Works well only for small kernels.

Note that this is not useful for regular grids, where direct
application of smoothing kernels is much faster. But the idea might be
kept to perform smoothing on irregular structures (meshes).

Author : Bertrand Thirion, 2006-2009
"""

import numpy as np
import nipy.neurospin.graph.field as ff

def cartesian_smoothing(ijk,data,sigma):
	"""
	Smoothing data on a(n uncomplete) cartesian grid
	
    Parameters
    ----------
	ijk : array od shape(nvox,3) list of the positions
        it is typically returned by (array(np.where())).T
	data :array of shape (ijk.shape[0],d) where d is the datas dimension 
         data sampled on the grid 
	sigma : the kernel parameter
	
    Returns
    -------
	data, which is the smoothed data

    fixme : in-place change of data ?
	"""
	if ijk.shape[1]!=3:
		raise ValueError, "please provide a (n,3) position array"
	ijk = ijk.astype(np.int)
	n = ijk.shape[0]
	if data.shape[0]!=n:
		raise ValueError, "incompatible dimension for data and ijk"
	data = data.astype(np.float)
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
    """
    fixme : comment and put elsewhere
    """
    from nipy.io.imageformats import load, save, Nifti1Image 
	offset = 1.0
	# generate some data
	x = offset*np.ones((30,30,30))
	ijk = np.where(np.ones((30,30,30)))
	ref = np.array([15,15,15])
	x[ref[0],ref[1],ref[2]]+=1;
	sigma = 1.87

	# take the ideal result
	dx = np.sum((ijk.T-ref)*(ijk.T-ref),1)
	target = 1/(2*np.pi*sigma**2)**(1.5) * np.exp(-dx/(2*sigma**2))
	target = target/np.sum(target)
	target += offset
	gaussian = x.copy()
	gaussian[ijk] = target
	save(Nifti1Image(gaussian,np.eye(4)),"/tmp/target.nii")

	# compute the result with the cartesian_smoothing procedure
	data = np.reshape(x.copy(),(np.size(x),1))
	trial = x.copy()
	trial[ijk] = np.squeeze(cartesian_smoothing(ijk.T,data,sigma))
	save(Nifti1Image(trial,np.eye(4)),"/tmp/result.nii")
	
	dt = gaussian-trial
	print "error:", (dt*dt).sum()
	return gaussian, trial

