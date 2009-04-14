import numpy as np
import os.path
import nifti
from fff2.spatial_models.parcellation import Parcellation


def parcel_input(Mask_Images,nbeta,learning_images,ths = .5,fdim=3,sform=None):
	"""
	MXYZ,mask,stats,Talairach,functional,stats = parcel_input(Mask_Images,nbeta,learning_images,ths = .5,fdim=3)
	INPUT:
	- Mask_Images: list of the paths of the mask images that are used
	to define the common space. These can be cortex segmentations
	(! resampled at the same resolution as the remainder of the data !)
	- nbeta: list of ids (integers) of the contrast of under study
	- learning_images: path of functional images used as input to the
	parcellation procesure. normally these are statistical images.
	- ths=.5: threshold to select the regions that are common across subjects.
	if ths = .5, thethreshold is half the number of subjects
	- sForm provides the transformation to Talairach space.
	if sform==None, this is taken from the image header
	OUTPUT:
	- pa a parcellation structure that essentially stores the individual masks and the
	grif coordinates
	- Stats: nbsubject-long list of arrays of shape
	(number of within-mask voxels of each subjet,fdim)
	which contains the amount of functional information
	available to parcel the data
	- Talairach: array of size (nbvoxels,3): MNI coordinates of the
	points corresponding to MXYZ

	NOTE:
	- In the future, the input should be reduced to a list of
	SPM.mat-like structures and a list of contrasts: all the paths
	would be straightforwardly inferred from this structures.
	"""
	Sess = len(Mask_Images)
	
	# Read the referential
	nim = nifti.NiftiImage(Mask_Images[0])
	ref_dim = tuple(nim.header['dim'][1:4])
	grid_size = np.prod(ref_dim)
	if sform==None:
		sform = nim.header['sform']
	
	# take the individual masks
	mask = []
	for s in range(Sess):
		nim = nifti.NiftiImage(Mask_Images[s])
		temp = np.transpose(nim.asarray())
		rbeta = nifti.NiftiImage(learning_images[s][0])
		maskb = np.transpose(rbeta.asarray())
		temp = np.minimum(temp,1-(maskb==0))		
		mask.append(temp)
		
	mask = np.squeeze(np.array(mask))

	# "intersect" the masks
	if ths ==.5:
		ths = Sess/2
	else:
		ths = np.minimum(np.maximum(ths,0),Sess-1)

	mask = mask>0
	smask = np.sum(mask,0)>ths
	MXYZ = np.array(np.where(smask))
	inmask = MXYZ.shape[1]
	mask = np.transpose(mask[:,MXYZ[0,:],MXYZ[1,:],MXYZ[2,:]])
	
	# Compute the position of each voxel in the common space	
	xyz = np.transpose(MXYZ.copy())
	nbvox = np.shape(xyz)[0]
	xyz = np.hstack((xyz,np.ones((nbvox,1))))
	Talairach = np.dot(xyz,np.transpose(sform))[:,:3]
		
	# Load the functional data
	Stats = []
	for s in range(Sess): 
		stat = []
		lXYZ = np.array(MXYZ[:,mask[:,s]])

		for B in range(nbeta):
			# the stats (noise-normalized contrasts) images
			rbeta = nifti.NiftiImage(learning_images[s][B])
			temp = np.transpose(rbeta.asarray())
			temp = temp[lXYZ[0,:],lXYZ[1,:],lXYZ[2,:]]
			temp = np.reshape(temp, np.size(temp))
			stat.append(temp)

		stat = np.array(stat)
		Stats.append(np.transpose(stat))
	
	# Possibly reduce the dimension of the  functional data
	if fdim<Stats[0].shape[1]:
		stats = np.concatenate(Stats)
		stats = np.reshape(stats,(stats.shape[0],nbeta))
		stats = stats-np.mean(stats)
		import numpy.linalg as L
		M1,M2,M3 = L.svd(stats,0)
		stats = np.dot(M1,np.diag(M2))
		stats = stats[:,:fdim]
		subj = np.concatenate([s*np.ones(Stats[s].shape[0]) for s in range(Sess)])
		Stats = [stats[subj==s] for s in range (Sess)]

	pa = Parcellation(1,np.transpose(MXYZ),mask-1)  
 	
	return pa,Stats,Talairach

def Parcellation_output(Pa,Mask_Images,learning_images,Talairach,nbru,verbose=1,swd = "/tmp"):
	"""
	Pa=Parcellation_output(Pa,Mask_Images,learning_images,Talairach,verbose=1,swd
	= '/tmp')
	Function that produces images that describe the spatial structure
	of the parcellation.  It mainly produces label images at the group
	and subject level
	INPUT:
	- Pa : the strcuture that describes the parcellation
	- Mask_Images: list of images that define the mask
	[NB: simply to make a template of the label images; 1 image would be enough]
	-learning_images: list of float images containing the input data
	[NB: simply to make a template of the label images; 1 image would be enough]
	- Talairach: array of shape (nbvox,3) that contains(approximated)
	MNI-coordinates of the brain mask voxels considered in the
	parcellation process
	- nbru: list of the names of the subjects
	- verbose=1 : verbosity level
	-swd = '/tmp' : where everything shall be written
	OUTPUT:
	- Pa: the updated strcuture that describes the parcellation
	"""
	Sess = Pa.nb_subj
	MXYZ = Pa.ijk
	Pa.set_subjects(nbru)
	
	# write the template image
	tlabs = Pa.group_labels
	LabelImage = os.path.join(swd,"template_parcel.nii") 
	rmask = nifti.NiftiImage(Mask_Images[0])
	ref_dim = tuple(rmask.header['dim'][1:4])
	grid_size = np.prod(ref_dim)
	
	Label = np.zeros(ref_dim)
	Label[Pa.ijk[:,0],Pa.ijk[:,1],Pa.ijk[:,2]]=tlabs+1
	nim = nifti.NiftiImage(np.transpose(Label),rmask.header)
	nim.description = 'group_level Label image obtained from a parcellation procedure'
	nim.save(LabelImage)

	# write subject-related stuff
	Jac = []
	if Pa.isfield('jacobian'):
		Jac = Pa.get_feature('jacobian')
		Jac = np.reshape(Jac,(Pa.k,Sess))
		
	for s in range(Sess):
		# write the images
		labs = Pa.label[:,s]
		LabelImage = os.path.join(swd,"parcel%s.nii" % nbru[s])
		JacobImage = os.path.join(swd,"jacob%s.nii" % nbru[s])		

		Label = np.zeros(ref_dim).astype('i')
		Label[Pa.ijk[:,0],Pa.ijk[:,1],Pa.ijk[:,2]]=labs+1
		nim = nifti.NiftiImage(np.transpose(Label),rmask.header)
		nim.description = 'individual Label image obtained from a parcellation procedure'
		nim.save(LabelImage)
		

		if ((verbose)&(np.size(Jac)>0)):
			Label = np.zeros(ref_dim)
			Label[Pa.ijk[:,0],Pa.ijk[:,1],Pa.ijk[:,2]]=Jac[labs,s]
			nim = nifti.NiftiImage(np.transpose(Label),rmask.header)
			nim.description = 'image of the jacobian of the deformation associated with the parcellation'
			nim.save(JacobImage)
		
	return Pa

def Parcellation_based_analysis(Pa,test_images,numbeta,swd = "/tmp",DMtx=None,verbose=1,method_id = 0):
	"""
	Pa = Parcellation_based_analysis(Pa,test_images,nbeta,swd = '/tmp',DMtx=None,verbose=1,method_id = 0)
	This function computes parcel averages and RFX at the parcel-level

	INPUT
	- Pa:  a desciptor of the parcel structure which is appended with some new information
	- test_images: path of functional images used as input to for
	inference. normally these are contrast images.
	double list is number of subjects and number of dimensions/contrasts
	- numbeta: list of the associated ids
	- swd=/tmp: where the results are to be written
	- DMtx = None possibly a design matrix for second-level analyses (not implemented yet)
	- verbose=1 : verbosity level
	- method_id = 0: an id of the method used.
	This is useful to compare the outcome of different Parcellation+RFX  procedures
	OUTPUT:
	- Pa: the update parcellation structure
	"""
	Sess = Pa.nb_subj
	MXYZ = np.transpose(Pa.ijk)
	mask = Pa.label>-1
	nbeta = len(numbeta)
	
	# 1. read the test data
	# TODO: Check that everybody is in the same referential
	Test = []
	for s in range(Sess):
		beta = []
		lXYZ = MXYZ[:,mask[:,s]]
		lXYZ = np.array(lXYZ)

		for B in range(nbeta):
			# the raw contrast images
			rbeta = nifti.NiftiImage(test_images[s][B])
			temp = np.transpose(rbeta.asarray())
			temp = temp[lXYZ[0,:],lXYZ[1,:],lXYZ[2,:]]
			temp = np.reshape(temp, np.size(temp))
			beta.append(temp)
			temp[np.isnan(temp)]=0 ##

		beta = np.array(beta)
		Test.append(np.transpose(beta))	

	# 2. compute the parcel-based stuff
	# and make inference inference (RFX,...)

	prfx = np.zeros((Pa.k,nbeta))
	vinter = np.zeros(nbeta)
	for B in range(nbeta):
		unitest = [np.reshape(Test[s][:,B],(np.size(Test[s][:,B]),1)) for s in range(Sess)]
		cname = 'contrast_%04d'%(numbeta[B])
		Pa.make_feature(unitest, cname)
		prfx[:,B] =  np.reshape(Pa.PRFX(cname,1),Pa.k)
		vinter[B] = Pa.variance_inter(cname)

	vintra = Pa.variance_intra(Test)

	if verbose:
		print vintra
		print vinter.mean()
			
	# 3. Write the stuff
	# write RFX images
	ref_dim = tuple(rbeta.header['dim'][1:4])
	grid_size = np.prod(ref_dim)
	tlabs = Pa.group_labels

	# write the prfx images
	for b in range(len(numbeta)):
		RfxImage = os.path.join(swd,"prfx_%s_%d.nii" % (numbeta[b],method_id))
		if ((verbose)&(np.size(prfx)>0)):
			print ref_dim
			rfx_map = np.zeros(ref_dim)
			rfx_map[Pa.ijk[:,0],Pa.ijk[:,1],Pa.ijk[:,2]] =  prfx[tlabs,b]
			nifti.NiftiImage(np.transpose(rfx_map),rbeta.header).save(RfxImage)
		
	return Pa


