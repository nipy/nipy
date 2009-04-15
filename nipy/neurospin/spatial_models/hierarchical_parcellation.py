import numpy as np

import fff2.graph.graph as fg
import fff2.graph.field as ff
import fff2.clustering.clustering as fc
import fff2.spatial_models.parcellation as fp

def _Field_Gradient_Jac(ref,target):
	"""
	Given a reference field ref and a target field target
	compute the jacobian of the target with respect to ref
	INPUT
	- ref in ff.Field instance
	- target is an array with similar shape as ref.field
	OUTPUT
	- fgj: array of shape (ref.V) that gives the jacobian
	implied by the ref.field->target transformation.
	"""
	import numpy.linalg as L
	n = ref.V
	xyz = ref.field
	dim = xyz.shape[1]
	fd = target.shape[1]
	fgj = []
	ln = ref.list_of_neighbors() 
	for i in range(n):
		j = ln[i]
		if np.size(j)>dim-1:
			dx = np.squeeze(xyz[j,:]-xyz[i,:])
			df = np.squeeze(target[j,:]-target[i,:])
			FG = np.dot(L.pinv(dx),df)
			fgj.append(L.det(FG))
		else:
			print i,j
			fgj.append(1)

	fgj = np.array(fgj)
	return fgj

def _exclusion_map_dep(i,ref,target, targeti):
	"""
	ancillary function to determin admissible values of some position
	within some predefined values
	INPUT:
	- i (int): index of the structure under consideration
	- ref: field that represent the topological structure of parcels
	and their standard position
	- target= array of shape (ref.V,3): current posistion of the parcels
	- targeti array of shape (n,3): possible new positions for the ith item
	OUPUT:
	- emap: aray of shape (n): a potential that yields the fitness
	of the proposed positions given the current configuration
	- rmin (double): ancillary parameter
	"""
	n = ref.V
	xyz = ref.field
	fd = target.shape[1]
	fgj = []
	ln = ref.list_of_neighbors() 
	j = ln[i]
	if np.size(j)>0:
		dx = xyz[j,:]-xyz[i,:]
		dx = np.squeeze(dx)
		rmin = np.min(np.sum(dx**2,1))/4
		u0 = xyz[i]+ np.mean(target[j,:]-xyz[j,:],1)
		emap = - np.sum((targeti-u0)**2,1)+rmin
	else:
		emap = np.zeros(targeti.shape[0])
	return emap

def _exclusion_map(i,ref,target, targeti):
	"""
	ancillary function to determin admissible values of some position
	within some predefined values
	INPUT:
	- i (int): index of the structure under consideration
	- ref: field that represent the topological structure of parcels
	and their standard position
	- target= array of shape (ref.V,3): current posistion of the parcels
	- targeti array of shape (n,3): possible new positions for the ith item
	OUPUT:
	- emap: aray of shape (n): a potential that yields the fitness
	of the proposed positions given the current configuration
	- rmin (double): ancillary parameter
	"""
	n = ref.V
	xyz = ref.field
	fd = target.shape[1]
	fgj = []
	ln = ref.list_of_neighbors() 
	j = ln[i]
	j = np.reshape(j,np.size(j))
	rmin = 0
	if np.size(j)>0:
		dx = xyz[j,:]-xyz[i,:]
		dx = np.reshape(dx,(np.size(j),fd))
		rmin = np.mean(np.sum(dx**2,1))/4
		u0 = xyz[i,:]+ np.mean(target[j,:]-xyz[j,:],0)
		emap = rmin - np.sum((targeti-u0)**2,1)
		for k in j:
			amap = np.sum((targeti-target[k,:])**2,1)-rmin/4
			emap[amap<0] = amap[amap<0]
	else:
		emap = np.zeros(targeti.shape[0])
	return emap,rmin

def _Field_Gradient_Jac_Map_(i,ref,target, targeti):
	"""
	Given a reference field ref and a target field target
	compute the jacobian of the target with respect to ref
	
	"""
	import numpy.linalg as L
	from numpy.random import randn
	n = ref.V
	xyz = ref.field
	fd = target.shape[1]
	fgj = []
	ln = ref.list_of_neighbors() 
	j = ln[i]
	if np.size(j)>0:
		print xyz[j,:]
		dx = xyz[j,:]-xyz[i,:]
		dx = np.squeeze(dx)
		idx = L.pinv(dx)
		for k in range(targeti.shape[0]):
			df = target[j,:]-targeti[k,:]
			df = np.squeeze(df)
			FG = np.dot(idx,df)
			fgj.append(L.det(FG))
	else:
		fgj = np.zeros(targeti.shape[0])

	fgj = np.array(fgj)
	return fgj

def _Field_Gradient_Jac_Map(i,ref,target, targeti):
	"""
	Given a reference field ref and a target field target
	compute the jacobian of the target with respect to ref
	
	"""
	import numpy.linalg as L
	n = ref.V
	xyz = ref.field
	fd = target.shape[1]
	fgj = []
	ln = ref.list_of_neighbors() 
	j = ln[i]
	if np.size(j)>0:
		dx = xyz[j,:]-xyz[i,:]
		dx = np.squeeze(dx)
		idx = L.pinv(dx)
		for k in range(targeti.shape[0]):
			df = target[j,:]-targeti[k,:]
			df = np.squeeze(df)
			FG = np.dot(idx,df)
			fgj.append(L.det(FG))
		fgj = np.array(fgj)

		for ij in np.squeeze(j):
			aux = []
			#print ij
			jj = np.squeeze(ln[ij])
			dx = xyz[jj,:]-xyz[ij,:]
			dx = np.squeeze(dx)
			idx = L.pinv(dx)
			ji = np.nonzero(jj==i)
			#print i, jj, ji
			for k in range(targeti.shape[0]):	
				df = target[jj,:]-target[ij,:]
				df[ji,:] = targeti[k,:]-target[ij,:]
				df = np.squeeze(df)
				FG = np.dot(idx,df)
				aux.append(L.det(FG))
			aux = np.array(aux)
			fgj = np.minimum(fgj,aux)
	else:
		fgj = np.zeros(targeti.shape[0])

	
	return fgj

def optim_hparcel(Ranat,RFeature, Feature,Pa, Gs,anat_coord,lamb=1.,dmax=10.,chunksize=1.e5,  niter=5,verbose=0):
	"""
	Core function of the heirrachical parcellation procedure.
	INPUT
	- Ranat: array of shape (n,3): set of positions sampled form the data
	- RFeature: array of shape (n,f): assocaited feature
	- Feature: list of subject-related feature arrays
	- Pa : parcellsation structure
	- Gs: graph that represents the topology of the parcellation
	- anat_coord: arrao of shape (nvox,3) space defining set of coordinates
	- lamb=1.0: parameter to weight position
	and feature impact on the algorithm
	- dmax = 10: locality parameter (in the space of anat_coord)
	to limit surch volume (CPU save)
	- chunksize=1.e5: note used here (to be removed)
	- niter = 5: number of iterations in teh algorithm
	- verbose=0: verbosity level
	OUTPUT
	- U: lsit of arrays that represent subject-depebndent parcellations
	- Proto_anat: array of shape (nvox) labelling of the common space
	(template parcellation)
	NOTE:
	to be worked out
	"""
	Sess = Pa.nb_subj
	# Ranat,RFeature,Pa,chunksize,dmax,lamb,Gs
	# a1. perform a rough clustering of the data to make prototype
	Labs = np.zeros(RFeature.shape[0])
	proto, Labs, J = fc.cmeans(RFeature,Pa.k,Labs,10)
	proto_anat = [np.mean(Ranat[Labs==k],0) for k in range(Pa.k)]
	proto_anat = np.array(proto_anat)
	proto = [np.mean(RFeature[Labs==k],0) for k in range(Pa.k)]
	proto = np.array(proto)
	proto_anat_old = proto_anat.copy()

	# a2. topological model of the parcellation
	# group-level part
	spatial_proto = ff.Field(Pa.k)
	spatial_proto.set_field(proto_anat)
	spatial_proto.Voronoi_diagram(proto_anat,anat_coord)
	spatial_proto.set_gaussian(proto_anat)
	spatial_proto.normalize()
	for git in range(niter):
		LP = []
		LPA = []
		U = []
		Energy = 0
		for s in range(Sess):			
			# b.subject-specific instances of the model
			# b.0 subject-specific information
			Fs = Feature[s]
			lac = anat_coord[Pa.label[:,s]>-1]
			target = proto_anat.copy()
			#print s
	
			for nit in range(1):
				lseeds = np.zeros(Pa.k,'i')
				aux = np.argsort(rand(Pa.k))
				tata = 0
				toto = np.zeros(lac.shape[0])
				for j in range(Pa.k):
					# b.1 speed-up :only take a small ball
					i = aux[j]
					dX = lac-target[i,:]
					iz = np.nonzero(np.sum(dX**2,1)<dmax**2)
					iz = np.reshape(iz,np.size(iz))
					if np.size(iz)==0:
						iz  = np.array([np.argmin(np.sum(dX**2,1))])
					
					# b.2: anatomical constraints
					lanat = np.reshape(lac[iz,:],(np.size(iz),anat_coord.shape[1]))
					pot = np.zeros(np.size(iz))
					#JM = _Field_Gradient_Jac_Map(i,spatial_proto,target,lanat)
					#pot[JM<=0] = np.infty
					#pot[JM>0] = -np.log(JM[JM>0])
					#rmin = 0
					#print np.shape(spatial_proto),np.shape(target),np.shape(lanat)
					JM,rmin = _exclusion_map(i,spatial_proto,target,lanat)
					pot[JM<0] = np.infty
					pot[JM>=0] = -JM[JM>=0]
					
					# b.3: add feature discrepancy
					dF = Fs[iz]-proto[i]
					dF = np.reshape(dF,(np.size(iz),proto.shape[1]))
					pot += lamb*np.sum(dF**2,1)
					
					# b.4: solution
					pb = 0
					if np.sum(np.isinf(pot))==np.size(pot):
						pot = np.sum(dX[iz,:]**2,1)
						tata +=1
						pb = 1

					#print np.size(iz),np.shape(iz),iz,np.argmin(pot)
					sol = iz[np.argmin(pot)]
					target[i] = lac[sol]
					
					if toto[sol]==1:
						print "pb",pb
						ln = spatial_proto.list_of_neighbors()
						argtoto = np.squeeze(np.nonzero(lseeds==sol))
						print i,argtoto,ln[i]
						if np.size(argtoto)==1: print [ln[argtoto]]
						print target[argtoto]
						print target[i]
						print rmin
						print JM[iz==lseeds[argtoto]]
						print pot[iz==lseeds[argtoto]]
					lseeds[i]= sol
					toto[sol]=1

				if verbose>1:
					jm = _Field_Gradient_Jac(spatial_proto,target)
					print jm.min(),jm.max(),np.sum(toto>0),tata
			
			# c.subject-specific parcellation
			g = Gs[s]
			f = ff.Field(g.V,g.edges,g.weights,Fs)
			u = f.constrained_voronoi(lseeds)
			U.append(u)

			Energy += np.sum((Fs-proto[u])*(Fs-proto[u]))/np.sum(Pa.label[:,s]>-1)
			# recompute the prototypes
			# (average in subject s)
			lproto = [np.mean(Fs[u==k],0) for k in range(Pa.k)]
			lproto = np.array(lproto)
			lproto[np.isnan(lproto)] = proto[np.isnan(lproto)]
			
			lproto_anat = [np.mean(lac[u==k],0) for k in range(Pa.k)]
			lproto_anat = np.array(lproto_anat)
			lproto_anat[np.isnan(lproto_anat)] = proto_anat[np.isnan(lproto_anat)]
			
			LP.append(lproto)
			LPA.append(lproto_anat)

		# recompute the prototypes across subjects
		proto_mem = proto.copy()
		proto_anat_mem = proto_anat.copy()
		proto = np.mean(np.array(LP),0)
		proto_anat = np.mean(np.array(LPA),0)
		displ = np.sqrt(np.sum((proto_mem-proto)**2,1).max())
		if verbose:
			print 'energy',Energy, 'displacement',displ
			
		# recompute the topological model
		spatial_proto.set_field(proto_anat)
		spatial_proto.Voronoi_diagram(proto_anat,anat_coord)
		spatial_proto.set_gaussian(proto_anat)
		spatial_proto.normalize()

		if displ<1.e-4*dmax: break
	return U,proto_anat

def hparcel(Pa,ldata,anat_coord,nbperm=0,niter=5, mu=10.,dmax = 10., lamb = 100.0, chunksize = 1.e5,verbose=0):
	"""
	Function that performs the parcellation by optimizing the
	sinter-subject similarity while retaining the connectedness
	within subject and some consistency across subjects.
	INPUT:
	- Pa a Parcel structure that essentially contains the grid position information
	and the individual masks
	- anat_coord: array of shape(nbvox,3) which defines the position
	 of the grid points in some space
	- nbperm=0: the number of times the parcellation and prfx
	computation is performed on sign-swaped data
	- niter=10: number of iterations to obtain the convergence of the method
	information in the clustering algorithm
	OUTPUT:
	- Pa: the resulting parcellation structure appended with the labelling
	"""
	
	# a various parameters
	nn = 18
	nbvox = Pa.nbvox
	xyz = Pa.ijk
	Sess = Pa.nb_subj
	
	#mu = 10.0 # weight of anatomical information in the feature vector
	#chunksize = 1.e5
	#dmax = 10
	#lamb = 100.0
	
	Gs = []
	Feature = []
	RFeature = []
	Ranat = []
	# browse the data
	for s in range(Sess):
		lxyz = xyz[Pa.label[:,s]>-1].astype('i')
		lnvox = np.sum(Pa.label[:,s]>-1)
		lac = anat_coord[Pa.label[:,s]>-1]
		g = fg.WeightedGraph(lnvox)
		g.from_3d_grid(lxyz,nn)
		g.remove_trivial_edges()
		beta = np.reshape(ldata[s],(lnvox,ldata[s].shape[1]))
		feature = np.hstack((beta,mu*lac/(1.e-15+np.std(anat_coord,0))))
		Gs.append(g)
		Feature.append(feature)
		aux = np.argsort(rand(lnvox))[:np.minimum(chunksize/Sess,lnvox)]
		RFeature.append(feature[aux,:])
		Ranat.append(lac[aux,:])

	RFeature = np.concatenate(RFeature)
	Ranat = np.concatenate(Ranat)

	# main function
	U,proto_anat = optim_hparcel(Ranat,RFeature, Feature,Pa, Gs,anat_coord,lamb, dmax,niter=niter,verbose=verbose)

	# write the individual labelling
	Labels = -1*np.ones((nbvox,Sess)).astype('i')
	for s in range(Sess):
		Labels[Pa.label[:,s]>-1,s] = U[s]
		
	palc = Pa.label.copy()
	Pa.set_labels(Labels)
	if Pa.isfield('functional'): Pa.remove_feature('functional')
	Pa.make_feature(ldata,'functional')
	prfx = Pa.PRFX('functional')
		
	# compute the group-level labels 
	u = fc.voronoi(anat_coord,proto_anat)
	Pa.group_labels  = u
	if nbperm>0:
		Pb = fp.Parcellation(Pa.k,xyz,palc)
		prfx0 = perm_prfx(Pb,Gs,Feature, ldata,anat_coord,nbperm,niter,dmax, lamb, chunksize)
		return Pa,prfx0
	else:
		return Pa


def perm_prfx(Pa,Gs,F0, ldata,anat_coord,nbperm=100,niter=5,dmax = 10., lamb = 100.0, chunksize = 1.e5):
	"""
	caveat: assumes that the functional dimension is 1
	"""
	# permutations for the assesment of the results
	prfx0 = []
	adim = anat_coord.shape[1]
	Sess = Pa.nb_subj
	palc = Pa.label.copy()
	for q in range(nbperm):
		Feature = []
		RFeature = []
		Ranat = []
		sldata = []
		for s in range(Sess):
			feature = F0[s].copy()
			lnvox = np.sum(Pa.label[:,s]>-1)
			lac = anat_coord[Pa.label[:,s]>-1]
			swap = (rand()>0.5)*2-1
			feature[:,:-adim] = swap*feature[:,:-adim]
			sldata.append(swap*ldata[s])
			Feature.append(feature)
			aux = np.argsort(rand(lnvox))[:np.minimum(chunksize/Sess,lnvox)]
			RFeature.append(feature[aux,:])
			Ranat.append(lac[aux,:])
			
		RFeature = np.concatenate(RFeature)
		Ranat = np.concatenate(Ranat)	
		# optimization part
		U,proto_anat = optim_hparcel(Ranat,RFeature, Feature,Pa, Gs,anat_coord,lamb, dmax,niter=niter)
		
		Labels = -1*np.ones((Pa.nbvox,Sess)).astype('i')
		for s in range(Sess):
			Labels[Pa.label[:,s]>-1,s] = U[s]
		
		Pa.set_labels(Labels)
		if Pa.isfield('functional'): Pa.remove_feature('functional')
		Pa.make_feature(sldata,'functional')
		prfx = Pa.PRFX('functional')
		print q, prfx.max(0)
		prfx0.append(prfx.max(0))
		Pa.set_labels(palc)
	print prfx0
	return prfx0


def _test():
	"""
	low-level test function that uses a surrogate dataset
	supposedly available in a local 'surrogate_data.out' volume
	"""

	# ------------------------------------
	# Get the data (mask+functional image)
	# take several experimental conditions
	# time courses could be used instead
	import nifti
	import pickle
	dataset = pickle.load(open('surrogate_data.out','r'))
	nbparcel = 10
	
	# ------------------------------------
	# 2. Read the data
	# one mask and several contrast images (or time series images)
	
	Sess = dataset.shape[0]
	ref_dim = (dataset.shape[1],dataset.shape[2])
	xy = np.transpose(np.array(np.where(dataset[0])))
	nbvox = np.size(xy,0)
	xyz = np.hstack((xy,np.zeros((nbvox,1))))
	
	Beta= []
	for s in range(Sess):
		beta = np.reshape(dataset[s],(nbvox,1))
		Beta.append(beta)
		
	anat_coord = xy
	mask = np.ones((nbvox,Sess)).astype('bool')
	ldata = Beta
	Pa = fp.Parcellation(nbparcel,xyz,mask-1)
	Pa =  hparcel(Pa,ldata,anat_coord)
	
	# write the results
	Label =  np.array([np.reshape(Pa.label[:,s],(30,30)) for s in range(Sess)])
	Label = Label.astype('int16')
	nifti.NiftiImage(Label).save("/tmp/label.nii")
	
	prfx = Pa.PRFX('functional')
	prfx_image = np.reshape(prfx[Pa.group_labels],(30,30))
	nifti.NiftiImage(prfx_image).save("/tmp/prfx.nii")

