#autoindent

"""
Generic Parcellation class:
Contains all the items that define a multi-subject parcellation

Author : Bertrand Thirion, 2005-2008

TODO : add a method 'global field', i.e. non-subject-specific info
"""

import numpy as np

class Parcellation:
	"""
	This is the basic Parcellation class:
	It is defined discretely , i.e.
	the parcellation is an explicit function on the set of voxels
	(or equivalently a labelling)
	we explictly handle the case of multiple subjects,
	where the labelling varies with the subjects
	
	- k is the number of parcels/classes
	- ijk: array of shape(nbvoxels,anatomical_dimension)
	that represents the grid of voxels to be parcelled
	(the same for all subjects) 
	typically anatomical_dimension=3
	- referential
	rerpresents the image referential, resoltuion, position and size 
	this is expressed as an affine (4,4) transformation matrix
	- label is an (nbvox*subjects) array: nbvox is the number of voxels within the binary mask
	if the voxel is not labelled in a given subject, then the label is -1
	thus the label has integer values in [-1,k-1]
	- group_labels is a  labelling of the template
	- subjects=none is a list of ids of the subjects
	by default, is is set as range(self.nb_subj)
	"""
	
	def __init__(self, k, ijk, label, group_labels=None, referential = None, subjects = []):
		"""
		Constructor
		"""
		self.k = k
		self.ijk = ijk.astype('i')
		self.nbvox = ijk.shape[0]
		if np.size(ijk)==self.nbvox:
			ijk = np.reshape(ijk, (self.nbvox, 1))

		self.anatdim = ijk.shape[1]
		self.label = label.astype('i')
		if np.size(label)==self.nbvox:
			label = np.reshape(label,(self.nbvox,1))
			
		self.nb_subj = label.shape[1]
		
		if group_labels==None:
			self.group_labels = np.zeros(self.nbvox).astype('i')
		else:
			self.group_labels = group_labels
			
		if subjects==[]:
			self.subjects = range(self.nb_subj)
		else:
			self.subjects = subjects
		
		self.referential = referential
		
		self.features = []
		self.fids = []
		self.check()
		
	def check(self):
		"""
		Some sanity check on the arguments of the class
		"""
		if self.label.min()<-1:
			raise ValueError,"All labels must be >=-1"

		if (self.label.max()>self.k-1):
			raise ValueError, "all labels must be <",self.k

		if self.label.shape[0]!=self.nbvox:
			print self.ijk.shape[0], self.nbvox
			raise ValueError,"The mask does not coincide with the labelling"
	
		if np.size(self.group_labels) != self.nbvox:
			print  np.size(self.group_labels), self.nbvox
			raise ValueError,"group_label has not the correct size"
		else:
			self.group_labels = np.reshape(self.group_labels,self.nbvox)
		
		if len(self.subjects)!=self.nb_subj:
			print len(self.subjects), self.nb_subj
			raise ValueError,"The list of subjects \
				does not coincide with the number of subjects"

	def	copy(self):
		"""
		Pa = self.copy()
		copy method
		"""
		Pa = Parcellation(self.k, self.ijk, self.label.copy(),\
						  self.group_labels.copy(), self.referential, self.subjects)

		for fid,f in zip(self.fids,self.features):
			Pa.set_feature(f, fid)
		
		return Pa
	
	def empty_parcels(self):
		"""
		q = self.empty_parcels()
		returns the ids of all parcels that are empty
		"""
		q = [i for i in range(self.k) if i not in self.label]
		q = np.array(q)
		return q


	def population(self):
		"""
		pop = self.population()
		the population of parcellation is the number of voxels included in each parcel
		this function simply returns an array of shape (number of parcels, number of subjects)
		that contains the parcel population
		"""
		pop = np.zeros((self.k,self.nb_subj))
		for s in range(self.nb_subj):
			for i in range(self.k):
				pop[i,s] = np.size(np.nonzero(self.label[:,s]==i))
					
		return pop


	def set_group_labels(self,glabels):
		"""
		self.reset_group_labels(glabels)
		reset the group labels
		"""
		if np.size(np.size(glabels)==self.nbvox):
			self.group_labels = glabels
			self.check()
		else:
			raise ValueError,"Not the correct shape for glabel"

	def set_labels(self,label):
		"""	
		self.reset_labels(label)
		resets the label array of the class
		INPUT:
		label = array of shape(self.k,self.nb_subj)
		"""
		if (np.shape(label)==(self.nbvox,self.nb_subj)):
			self.label = label
			self.check()
		else:
			raise ValueError,"Not the correct shape for label"

	def set_subjects(self,subjects):
		"""
		self.reset_subjects(subjects)
		reset the list of subjects name
		INPUT:
		- subjects = a list of subjects id with length self.nb_subj
		"""
		if len(subjects)!=self.nb_subj:
			print len(subjects), self.nb_subj
			raise ValueError,"The list of subjects \
				does not coincide with the number of subjects"
		else:
			self.subjects = subjects
		
	
	def add_subjects(self,label,nsubj_id):
		"""
		self.add_subjects(label,subj_id)
		Add some subjects to the structure
		Not implemented yet.
		"""
		print "not implemented yet"
		pass


	#-----------------------------------------------------------
	#-- Parcel-level analysis of various information -----------
	#-----------------------------------------------------------

	def set_info(self,data,fid):
		"""
		self.set_info(data,fid):
		Add some non-subject specific feature information
		defined on a voxel-by voxel basis
		INPUT:
		- feature: an array of shape(self.nbvox,dim),
		where dim is the info dimension
		- fid : an identifier of the information
		"""
		pass

	def make_feature_from_info(self,fid):
		"""
		self.make_feature_from_info(fid)
		"""
		pass

	
	
		
		
	
	def set_feature(self,feature,fid):
		"""
		self.set_feature(feature,fid):
		Add a feature to the feature list of the structure
		INPUT:
		- feature: an array of shape(self.nb_subj,self.k,fdim),
		where fdim is the feature dimension
		- fid: a string that is the feature id
		"""
		# 1. test that the feature does not exist yet
		i = np.array([fid==f for f in self.fids])
		i = np.nonzero(i)
		i = np.reshape(i,np.size(i))
		if np.size(i)>0:
			raise ValueError,"Existing feature id"
			
		# 2. if no, add the new one
		if np.size(feature)==self.nbsubj*self.k:
			feature = np.reshape(feature,(self.nbsub,self.k))
		
		if (feature.shape[0])==self.nb_subj:
			if (feature.shape[1])==self.k:
				self.features.append(feature)
				self.fids.append(fid)
			else: raise ValueError,"incoherent size"
		else: raise ValueError,"incoherent size"
		
	def get_feature(self,fid):
		"""
		self.get_feature(fid):
		Get feature to the feature list of the structure
		INPUT:
		- fid: a string that is the feature id
		OUTPUT
		- feature: an array of shape(self.nb_subj,self.k,fdim),
		where fdim is the feature dimension
		"""
		i = np.array([fid==f for f in self.fids])
		i = np.nonzero(i)
		i = np.reshape(i,np.size(i))
		if np.size(i)==0:
			print "The feature does not exist"
			return None
		if np.size(i)==1:
			return self.features[int(i)]

	def isfield(self,fid):
		"""
		self.isfield(fid)
		tests whether fid is known as a field 
		"""
		i = np.array([fid==f for f in self.fids])
		i = np.nonzero(i)
		i = np.reshape(i,np.size(i))	
		return np.size(i)==1
	
	def remove_feature(self,fid):
		"""
		self.remove_feature(fid):
		Remove feature from the feature list of the structure
		INPUT:
		- fid: a string that is the feature id
		"""
		i = np.array([fid!=f for f in self.fids])
		i = np.nonzero(i)
		i = np.reshape(i,np.size(i))
		Rf = [self.features[j] for j in i]
		Rfid =[self.fids[j] for j in i]
		self.features = Rf
		self.fids= Rfid

	def make_feature(self,data, fid,subj=-1,method="average"):
		"""
		self.make_feature(data,fid,subj=-1,method='average'):
		Compute and Add a feature to the feature list of the structure
		INPUT:
		- data: a list of arrays of shape(nbvoxels,fdim),
		where fdim is the feature dimension
		NOTE: if subj>-1, then data is simply an array of shape (nbvoxels,fdim)
		- fid: a string that is the feature id
		- subj = -1: subject in which this is performed
		if subject==-1, this is in all subjects,
		and it is checked that the fid is not defined yet
		otherwise, this is in one particular subject,
		and the feature may be overriden
		- method = 'average', the way to compute the feature 
		"""
		# 1. test that the feature does not exist yet
		i = np.array([fid==f for f in self.fids])
		i = np.nonzero(i)
		i = np.reshape(i,np.size(i))
		if subj==-1:
			if np.size(i)>0:
				print fid,
				raise ValueError,"Existing feature id"

		    #2. If no, compute the new feature and add it to the list
			feature = self.average_feature(data)
			self.features.append(feature)
			self.fids.append(fid)
		else:
			if subj>self.nb_subj-1:
				raise ValueError,"incoherent subject index"
			if np.size(i)==0:
				# create the feature
				i = len(self.fids)
				self.fids.append(fid)
				self.features.append(np.zeros((self.nb_subj,self.k,data.shape[1])))			

			# check that the dimension is OK
			if data.shape[1]!=self.features[i].shape[2]:
				raise ValueError,"Incompatible feature dimension"

			# make the feature
			feature = self.average_feature(data,subj)
			self.features[i][subj]=feature
			


	def PRFX(self,fid,zstat=1,DMtx = None):
		"""
		RFX = self.PRFX(fid,zstat=1)
		Compute the Random effects of the feature on the parcels across subjects
		INPUT:
		- fid is the feature identifier;
		it is assumed that the feature is 1-dimensional
		-zstat indicator variable for the output variate
		if ztsat==0, the basic student statistic is returned
		if zstat==1, the student stat is converted to a normal(z) variate
		- DMtx = None : design matrix for the model.
		So far, it is assumed that DMtx = np.ones(self.nb_subj)
		OUPUT:
		- RFX: array with shape (self.k,fdim)
		containing the parcel-based RFX.
		"""
		if self.nb_subj<2:
			print "Sorry, there is only one subject"
			return []

        #1. find the field id
		i = np.array([fid==f for f in self.fids])
		i = np.nonzero(i)
		i = np.reshape(i,np.size(i))
		if np.size(i)==0:
			print "The feature does not exist"
			return []

		#2. Compute the PRFX
		PF = self.features[i]
		eps = 1.e-7
		SF = np.array(np.std(PF,0))
		SF = np.maximum(SF,eps)
		MF = np.array(np.mean(PF,0))
		RFX = MF/SF*np.sqrt(self.nb_subj)
		if zstat==1:
			import scipy.stats as ST
			pval = ST.t.cdf(RFX,self.nb_subj-1)
			RFX = np.minimum(10,np.maximum(-10,ST.norm.ppf(pval)))

		return RFX		

	def average_feature(self,Feature,subj=-1):
		"""
		PF = self.average_feature(Feature)
		compute parcel-based fetaure bu averaging voxel-based quantities
		INPUT:
		- Feature is a list of length self.nb_subj,
		so that for each s in 0..self.nb_subj-1, 
		that Feature[s] is an (nvox,fdim)-shaped array
		where nvox is the number of voxels in subject s with label >-1
		- subj = -1: subject in which this is performed
		if subj==-1, this is in all subjects,
		and it is checked that the fid is not defined yet
		if subj>-1, this is in one particular subject,
		and Feature merely is an array, not a list  
		OUPUT:
		- PF: array of shape (self.nb_subj,self.k,fdim) if subj==-1
		or (self.k,fdim)
		containing the parcel-based features.
		
		"""
		if subj==-1:
			# Do the computation in available subjects
			PF = []
			
			for s in range (self.nb_subj):
				pf = np.zeros((self.k, Feature[s].shape[1])).astype('d')
				pop = np.zeros((self.k))
				j = 0
				for i in range(self.nbvox):
					if self.label[i,s]>-1:
						pf[self.label[i,s],:] += Feature[s][j,:]
						j = j+1
						pop[self.label[i,s]] +=1

				for i in range(self.k):
					if pop[i]>0:
						pf[i,:]/=(pop[i])

				PF.append(pf)

			PF = np.array(PF)
		else:
			# Do the computation in subject s specifically
			if subj>self.nb_subj-1:
				raise ValueError,"incoherent subject index"
			s = subj
			PF = np.zeros((self.k, Feature.shape[1])).astype('d')
			pop = np.zeros((self.k))
			j = 0
			for i in range(self.nbvox):
				if self.label[i,s]>-1:
					PF[self.label[i,s],:] += Feature[j,:]
					j = j+1
					pop[self.label[i,s]] +=1

			for i in range(self.k):
				if pop[i]>0:
					PF[i,:]/=(pop[i])

		return(PF)

	def variance_inter(self,fid):
		"""
		HI = self.variance_inter(fid)
		Compute the variance of the feature at each parcel across subjects
		INPUT:
		- fid is the feature identifier
		OUPUT:
		- HI
		"""
		#.0 check that there is more than 1 subject
		if self.nb_subj<2:
			print "Sorry, there is only one subject"
			return []
		
		#1. find the field id
		i = np.array([fid==f for f in self.fids])
		i = np.nonzero(i)
		i = np.reshape(i,np.size(i))
		if np.size(i)==0:
			print "The feature does not exist"
			return []
		
		#2. Compute the corss-subject variance
		AF = self.features[i]
		pop = np.transpose(self.population())
		MAF = np.mean(AF,0)
		dAF = AF-MAF
		for d in range(AF.shape[2]):
			dAF[:,:,d] = (pop>0)*dAF[:,:,d]**2

		HI = np.sum(dAF)/np.transpose(np.repeat(np.sum(pop>0)-1,dAF.shape[2],1))
		return HI


	def var_feature_intra(self,Feature):
		"""
		compute the feature variance in each subject and each parcel
		"""
		VF = []
		for s in range (self.nb_subj):
			pf = np.zeros((self.k, Feature[s].shape[1])).astype('d')
			vf = np.zeros((self.k, Feature[s].shape[1])).astype('d')
			pop = np.zeros((self.k))
			j = 0
			for i in range(self.nbvox):
				if self.label[i,s]>-1:
					pf[self.label[i,s],:] += Feature[s][j,:]
					pop[self.label[i,s]] +=1
					j = j+1
			for i in range(self.k):
				if pop[i]>0:
					pf[i,:]/=(pop[i])
			j = 0
			for i in range(self.nbvox):
				if self.label[i,s]>-1:
					dfeat = pf[self.label[i,s],:] - Feature[s][j,:]
					vf[self.label[i,s],:] += dfeat*dfeat#np.dot(dfeat,dfeat)
					j = j+1
			for i in range(self.k):
				if pop[i]>1:
					vf[i,:]/=(pop[i]-1)
		
			VF.append(vf)

		return(VF)

	def variance_intra(self,data,bweight=0):
		"""
		Vintra = self.variance_intra(fid)
		Compute the variance of the feature at each parcel within each subject
		INPUT:
		- data is the data on which the  variance is estimated:
		this is a list of arrays
		- bweight=0: flag for the relative weighting of the parcels
		if bweight = 1, the variance of each parcels is weighted by its size
		else, all parcels are equally weighted
		OUPUT:
		- VA : array of shape (self.k) of the variance 
		"""
		
		VF = self.var_feature_intra(data)
		Vintra = np.zeros((self.nb_subj)).astype('d')
		if bweight==1:
			pop = self.population()
			for s in range (self.nb_subj):
				Vintra[s] = np.mean(np.mean(VF[s],1)*(pop[:,s]-1))/np.mean(pop[:,s])
		else:
			for s in range (self.nb_subj):
				Vintra[s] = np.mean(np.mean(VF[s]))

		return Vintra


	def boxplot_feature(self,pid,fids):
		"""
		self.show_feature(pid,fids)
		This function makes a boxplot of the feature distribution
		in a given parcel across subjects
		INPUT:
		- pid = parcel identifier an integer within the [0..self.K] range
		- fids = list of features of inetegers
		"""
		#1. test that pid is coorect
		if pid<0:
			raise ValueError,"Negative parcel id"
		if pid>self.k:
			raise ValueError,"Wrong parcel id"

		# 2. test that the feature(s) exist
		idx = []
		for fid in fids: 
			i = np.array([fid==f for f in self.fids])
			i = np.nonzero(i)
			i = np.reshape(i,np.size(i))
			if np.size(i)==0:
				raise ValueError,"The feature does not exist yet"
			idx.append(i)

		#3 get the data and make the figure
		dataplot = []
		for j in idx:
			dataplot.append(np.transpose(self.features[j][:,pid]))

		dataplot = np.transpose(np.concatenate(dataplot))
		print np.shape(dataplot)
		import matplotlib.pylab as mp
		mp.figure()
		mp.boxplot(dataplot)

