# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#autoindent

"""
Generic Parcellation class:
Contains all the items that define a multi-subject parcellation

Author : Bertrand Thirion, 2005-2008

TODO : add a method 'global field', i.e. non-subject-specific info
"""

import numpy as np
#import discrete_domain as dm

###################################################################
# new parcellation class
###################################################################



class MultiSubjectParcellation(object):
    """ 
    MultiSubjectParcellation class are used to represent parcels 
    that can have different spatial different contours 
    in a given group of subject
    It consists of 
    self.domain: the specification of a domain
    self.template_labels the specification of a template parcellation
    self.individual_labels the specification of individual parcellations
    
    fixme:should inherit from mroi.MultiROI
    """

    def __init__(self, domain, template_labels=None, individual_labels=None, 
                 nb_parcel=None):
        """
        Parameters
        ----------
        domain: discrete_domain.DiscreteDomain instance,
                definition of the space considered in the parcellation
        template_labels: array of shape domain.size, optional 
                         definition of the template labelling
        individual_labels: array of shape (domain.size, nb_subjects), optional,
                           the individual parcellations 
                           corresponding to the template 
        nb_parcel: int, optional,
                   number of parcels in the model
                   can be inferred as template_labels.max()+1, or 1 by default
                   cannot be smaller than template_labels.max()+1
        """
        self.domain = domain
        self.template_labels = template_labels
        self.individual_labels = individual_labels
        
        self.nb_parcel = 1
        if template_labels is not None:
            self.nb_parcel = template_labels.max()+1
        if nb_parcel is not None:
            self.nb_parcel = nb_parcel

        self.check()
        self.nb_subj = 0
        if individual_labels is not None:
            if individual_labels.shape[0] == individual_labels.size:
                self.individual_labels = individual_labels[:, np.newaxis]
            self.nb_subj = self.individual_labels.shape[1]
        
        self.features = {}
            
            
    def copy(self):
        """ Returns a copy of self
        """
        msp =  MultiSubjectParcellation(self.domain.copy(), 
                                        self.template_labels.copy(), 
                                        self.individual_labels.copy(),
                                        self.nb_parcel)
        
        for fid in self.features.keys():
            msp.set_feature(fid, self.get_feature(fid).copy())
        return msp

    def check(self):
        """ Performs an elementary check on self
        """
        size = self.domain.size
        if self.template_labels is not None:
            nvox = np.size(self.template_labels)
            if size!=nvox:
                raise ValueError, "template labels not consistent with domain"
        if self.individual_labels is not None:
            n2 = self.individual_labels.shape[0]
            if size != n2:
                raise ValueError, "Individual labels not consistent with domain"
        if self.nb_parcel < self.template_labels.max()+1:
            raise ValueError, "too many labels in template"
        if self.nb_parcel < self.individual_labels.max()+1:
            raise ValueError, "Too many labels in individual models"

    def set_template_labels(self, template_labels):
        """
        """
        self.template_labels = template_labels
        self.check()
        
    def set_individual_labels(self, individual_labels):
        """
        """
        self.individual_labels = individual_labels
        self.check()
        self.nb_subj = self.individual_labels.shape[1]

    def population(self):
        """ Returns the counting of labels per voxel per subject

        Returns
        -------
        population: array of shape (self.nb_parcel, self.nb_subj)
        """
        population = np.zeros((self.nb_parcel, self.nb_subj)).astype(np.int)
        for ns in range(self.nb_subj):
            for k in range(self.nb_parcel):
                population[k, ns] = np.sum(self.individual_labels[:, ns]==k)
        return population  

    def make_feature(self, fid, data):
        """ Compute parcel-level averages of data
        
        Parameters
        ----------
        fid: string, the feature identifier
        data: array of shape (self.domain.size, self.nb_subj, dim) or 
              (self.domain.sire, self.nb_subj)
              Some information at the voxel level
        
        Returns
        -------
        pfeature: array of shape(self.nb_parcel, self.nbsubj, dim)
                  the computed feature data
        """
        if len(data.shape)<2:
            raise ValueError, "Data array should at least have dimension 2"
        if len(data.shape)>3:    
            raise ValueError, "Data array should have <4 dimensions"
        if ((data.shape[0] != self.domain.size) or 
            (data.shape[1]!= self.nb_subj)):
            raise ValueError, 'incorrect feature size'
        
        if len(data.shape)==3:
            dim = data.shape[2]
            pfeature = np.zeros((self.nb_parcel, self.nb_subj, dim))
        else:
            pfeature = np.zeros((self.nb_parcel, self.nb_subj))
        
        for k in range(self.nb_parcel):
            for s in range(self.nb_subj):
                dsk = data[self.individual_labels[:, s]==k, s]
                pfeature[k, s] = np.mean(dsk, 0)
        
        self.set_feature(fid, pfeature)
        return feature


    def set_feature(self, fid, data):
        """
        Parameters
        ----------
        fid: string, the feature identifier
        data: array of shape (self.nb_parcel, self.nb_subj, dim) or 
              (self.nb_parcel, self.nb_subj)
              the data to be set as parcel- and subject-level information
        """
        if len(data.shape)<2:
            raise ValueError, "Data array should at least have dimension 2"
        if (data.shape[0] != self.nb_parcel) or (data.shape[1]!= self.nb_subj):
            raise ValueError, 'incorrect feature size'
        else:
            self.features.update({fid:data})
    
    def get_feature(self, fid):
        """
        Parameters
        ----------
        fid: string, the feature identifier
        """
        return self.features[fid]

            
###################################################################
# parcellation class
###################################################################

class Parcellation(object):
    """
    This is the basic Parcellation class:
    It is defined discretely , i.e.
    the parcellation is an explicit function on the set of voxels
    (or equivalently a labelling)
    we explictly handle the case of multiple subjects,
    where the labelling varies with the subjects

    k is the number of parcels/classes
    ijk: array of shape(nbvoxels,anatomical_dimension)
         that represents the grid of voxels to be parcelled
         (the same for all subjects) 
         typically anatomical_dimension=3
    referential rerpresents the image referential, 
                  resoltuion, position and size 
                  this is expressed as an affine (4,4) transformation matrix
    label (nbvox, subjects) array: nbvox is the number of voxels
            within the binary mask
            if the voxel is not labelled in a given subject, then the label is -1
            thus the label has integer values in [-1,k-1]
    group_labels is a  labelling of the template
    subjects=none is a list of ids of the subjects
                    by default, is is set as range(self.nb_subj)
    """
    
    def __init__(self, k, ijk, label, group_labels=None, 
                       referential = None, subjects = []):
        """
        Constructor
        """
        self.k = k
        self.ijk = ijk.astype(np.int)
        self.nbvox = ijk.shape[0]
        if np.size(ijk)==self.nbvox:
            ijk = np.reshape(ijk, (self.nbvox, 1))

        self.anatdim = ijk.shape[1]
        self.label = label.astype(np.int)
        if np.size(label)==self.nbvox:
            label = np.reshape(label,(self.nbvox,1))
            
        self.nb_subj = label.shape[1]
        
        if group_labels==None:
            self.group_labels = np.zeros(self.nbvox).astype(np.int)
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
            raise ValueError, "all labels must be < %d" %self.k

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

    def    copy(self):
        """
        Pa = self.copy()
        copy method
        """
        Pa = Parcellation(self.k, self.ijk, self.label.copy(),\
                          self.group_labels.copy(), self.referential, 
                          self.subjects)

        for fid,f in zip(self.fids,self.features):
            Pa.set_feature(f, fid)
        
        return Pa

    def population(self):
        """
        pop = self.population()
        the population of parcellation is the number of voxels 
        included in each parcel
        this function simply returns an array of shape 
        (number of parcels, number of subjects)
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
        resets the label array of the class
        
        Parameters
        ----------
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
        
        Parameters
        ----------
        subjects = a list of subjects id with length self.nb_subj
        """
        if len(subjects)!=self.nb_subj:
            print len(subjects), self.nb_subj
            raise ValueError,"The list of subjects \
                does not coincide with the number of subjects"
        else:
            self.subjects = subjects
        

    #-----------------------------------------------------------
    #-- Parcel-level analysis of various information -----------
    #-----------------------------------------------------------
    
    def set_feature(self,feature,fid):
        """
        self.set_feature(feature,fid):
        Add a feature to the feature list of the structure
        
        Parameters
        ----------
        feature: array of shape(self.nb_subj,self.k,fdim),
                 where fdim is the feature dimension
        fid, string, the feature id
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
        Get feature to the feature list of the structure
        
        Parameters
        ----------
        fid, string, the feature id
        
        Returns
        -------
        feature: array of shape(self.nb_subj,self.k,fdim),
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
        tests whether fid is known as a field 
        """
        i = np.array([fid==f for f in self.fids])
        i = np.nonzero(i)
        i = np.reshape(i,np.size(i))    
        return np.size(i)==1
    
    def remove_feature(self,fid):
        """
        Remove feature from the feature list of the structure
        
        Parameters
        ----------
        fid, string, the feature id
        """
        i = np.array([fid!=f for f in self.fids])
        i = np.nonzero(i)
        i = np.reshape(i,np.size(i))
        Rf = [self.features[j] for j in i]
        Rfid =[self.fids[j] for j in i]
        self.features = Rf
        self.fids= Rfid

    def make_feature(self, data, fid, subj=-1, method="average"):
        """
        Compute and Add a feature to the feature list of the structure
        
        Parameters
        ----------
        data: a list of arrays of shape(nbvoxels,fdim),
              where fdim is the feature dimension
              Note: if subj>-1, then data is simply an array 
              of shape (nbvoxels,fdim)
        fid, string, the feature id
        subj = -1: subject in which this is performed
             if subject==-1, this is in all subjects,
             and it is checked that the fid is not defined yet
             otherwise, this is in one particular subject,
             and the feature may be overriden
        method = 'average', the way to compute the feature 
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
                self.features.append(np.zeros((self.nb_subj, 
                               self.k, data.shape[1])))            

            # check that the dimension is OK
            if data.shape[1]!=self.features[i].shape[2]:
                raise ValueError,"Incompatible feature dimension"

            # make the feature
            feature = self.average_feature(data,subj)
            self.features[i][subj]=feature

    def PRFX(self, fid, zstat=1, DMtx = None):
        """
        Compute the Random effects of the feature on the 
        parcels across subjects
        
        Parameters
        ----------
        fid : str
           feature identifier; it is assumed that the feature is
           1-dimensional
        zstat : int
           indicator variable for the output variate. If ztsat==0, the
           basic student statistic is returned. If zstat==1, the
           student stat is converted to a normal(z) variate
        DMtx : None
           design matrix for the model.  So far, it is assumed that DMtx
           = np.ones(self.nb_subj)
        
        Returns
        -------
        RFX: array with shape (self.k,fdim)
           the parcel-based RFX.
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
        compute parcel-based fetaure bu averaging voxel-based quantities

        Parameters
        ----------
        Feature is a list of length self.nb_subj,
                so that for each s in 0..self.nb_subj-1, 
                that Feature[s] is an (nvox,fdim)-shaped array
                where nvox is the number of voxels in subject s with label >-1
        subj = -1: subject in which this is performed
             if subj==-1, this is in all subjects,
             and it is checked that the fid is not defined yet
             if subj>-1, this is in one particular subject,
             and Feature merely is an array, not a list  
        
        Returns
        -------
        PF: array of shape (self.nb_subj,self.k,fdim) if subj==-1
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
        Compute the variance of the feature at each parcel across subjects
        
        Parameters
        ----------
        fid, string, the feature identifier
        
        Returns
        -------
        HI, array of shape (self.k) (?) 
            the inter-subject variance
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

        HI = np.sum(dAF)/ np.repeat(np.sum(pop>0)-1,dAF.shape[2],1).T
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
                    vf[self.label[i,s],:] += dfeat*dfeat
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
        
        Parameters
        ----------
        data is the data on which the  variance is estimated:
             this is a list of arrays
        bweight=0: flag for the relative weighting of the parcels
                   if bweight = 1, the variance of each parcels 
                   is weighted by its size
                   else, all parcels are equally weighted
        
        Returns
        -------
        VA : array of shape (self.k) of the variance 
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
        
        Parameters
        ----------
        pid = parcel identifier an integer within the [0..self.K] range
        fids = list of features of inetegers
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

