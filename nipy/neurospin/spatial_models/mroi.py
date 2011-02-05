# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np

from nibabel import load, save, Nifti1Image 

import discrete_domain as ddom

###############################################################################
# class MultiROI
###############################################################################


class MultiROI(object):
    """
    This is an abstract class from which different types of
    multiple ROI classes will be derived
    """

    def __init__(self, domain, k, rid=''):
        """
        Parameters
        ----------
        domain: ROI instance
                defines the spatial context of the SubDomains
        k: non-negative int, number of regions considered
        id: string, optional, identifier
        """
        self.domain = domain
        self.k = k
        self.id = rid
        self.roi_features = {}

    def set_roi_feature(self, fid, data):
        """
        """
        if len(data) != self.k:
            raise ValueError, 'data should have length k'
        self.roi_features.update({fid:data})

    def get_roi_feature(self, fid):
        return self.roi_features[fid]
        
    
    def select(self, valid):
        """Select a subset of ROIs and he associated features
        """
        if valid.size != self.k:
            raise ValueError, 'Invalid size for valid'

        for k in range(self.k):
            if valid[k]==0:
                self.label[self.label==k] = -1
            
        oldk = self.k
        self.k = valid.sum()
                
        for fid in self.roi_features.keys():
            f = self.roi_features.pop(fid)
            sf = np.array([f[k] for k in range(oldk) if valid[k]])
            self.set_roi_feature(fid, sf)
            
          
###############################################################################
# class SubDomains
###############################################################################


class SubDomains(object):
    """
    This is another implementation of Multiple ROI,
    where the reference to a given domain is explicit

    fixme : make roi_features implementation consistent
    """

    def __init__(self, domain, label, id='', no_empty_label=True):
        """
        Parameters
        ----------
        domain: ROI instance
                defines the spatial context of the SubDomains
        label: array of shape (domain.size), dtype=np.int,
               the label values greater than -1 correspond to subregions
               labelling
        id: string, optional, identifier
        no_empty_label: Bool, optional
                         if True absent label values are collapsed
                        otherwise, label is kept but empty regions might exist
        """
        # check that label is consistant with domain
        if np.size(label)!= domain.size:
            raise ValueError, 'inconsistent labels and domains specification'
        self.domain = domain
        self.label = np.reshape(label, label.size).astype(np.int)

        if no_empty_label:
            # remove labels with no discrete element
            lmap = np.unique(label[label>-1])
            for i,k in enumerate(lmap):
                self.label[self.label==k] = i 
            
        # number or ROIs : number of labels>-1
        self.k = self.label.max()+1
        self.size = np.array([np.sum(self.label==k) for k in range(self.k)])

        #id
        self.id = id
        
        # initialize empty feature/roi_feature dictionaries
        self.features = {}
        self.roi_features = {}
        


    def copy(self, id=''):
        """ Returns a copy of self
        Note that self.domain is not copied
        """
        cp = SubDomains( self.domain, self.label.copy(), id=id )
        for fid in self.features.keys():
            f = self.features[fid]
            sf = [f[k].copy() for k in range(self.k)]
            cp.set_feature(fid, sf)
        for fid in self.roi_features.keys():
            cp.set_roi_feature(fid, self.roi_features[fid].copy())
        return cp
        
    def get_coord(self, k):
        """ returns self.coord[k]
        """
        if k>self.k-1:
            raise ValueError, 'works only for k<%d'%self.k
        return self.domain.coord[self.label.k]

    def get_size(self):
        """ returns size, k-length array
        """
        return self.size

    def get_volume(self, k):
        """ returns self.local_volume[k]
        """
        if k>self.k-1:
            raise ValueError, 'works only for k<%d'%self.k
        return self.domain.local_volume[self.label==k]

    def select(self, valid, id='', auto=True, no_empty_label=True):
        """
        returns an instance of multiple_ROI
        with only the subset of ROIs for which valid

        Parameters
        ----------
        valid: array of shape (self.k),
               which ROIs will be included in the output
        id: string, optional,
            identifier of the output instance
        auto: bool, optional,
              if True then self = self.select()
        no_empty_label: bool, optional,
                        if True, labels with no matching site are excluded

        Returns
        -------
        the instance, if auto==False, nothing otherwise
        """
        if valid.size != self.k:
            raise ValueError, 'Invalid size for valid'

        if auto:
            for k in range(self.k):
                if valid[k]==0:
                    self.label[self.label==k] = -1
            self.id = id
            if no_empty_label:
                lmap = np.unique(self.label[self.label>-1])
                for i,k in enumerate(lmap):
                    self.label[self.label==k] = i 

                oldk = self.k
                # number or ROIs : number of labels>-1
                self.k = self.label.max()+1
                self.size = np.array([np.sum(self.label==k)
                                      for k in range(self.k)])
                
                for fid in self.features.keys():
                    f = self.features.pop(fid)
                    sf = [f[k] for k in range(oldk) if valid[k]]
                    self.set_feature(fid, sf)
                    
                for fid in self.roi_features.keys():
                    f = self.roi_features.pop(fid)
                    sf = np.array([f[k] for k in range(oldk) if valid[k]])
                    self.set_roi_feature(fid, sf)
            
            else:
                for fid in self.features.keys():
                    f = self.features(fid)
                    for k in range(self.k):
                        if valid[k]==0:
                            f[k] = np.array([])
                            
                #for fid in self.roi_features.keys():
                #    f = self.roi_features(fid)
                #    for k in range(self.k):
                #        if valid[k]==0:
                #            f[k] = np.array([])
                            
            self.size = np.array([np.sum(self.label==k)
                                  for k in range(self.k)])
        else:
            label = -np.ones(self.domain.size)
            remap = np.arange(self.k)
            remap[valid==0] = -1
            label[self.label>-1] = remap[self.label[self.label>-1]]
            SD = SubDomains(self.domain, label, no_empty_label=no_empty_label)
            return SD

    def make_feature(self, fid, data, override=True):
        """
        Extract a set of ffeatures from a domain map

        Parameters
        ----------
        fid: string,
             feature identifier
        data: array of shape(deomain.size) or (domain, size, dim),
              domain map from which ROI features are axtracted
        override: bool, optional,
                  Allow feature overriding 
        """
        if data.shape[0] != self.domain.size:
            raise ValueError, "Incorrect data provided"
        dat = [data[self.label==k] for k in range(self.k)]
        self.set_feature(fid, dat, override)
    
    
    def set_feature(self, fid, data, override=True):
        """
        Append a feature 'fid'
        
        Parameters
        ----------
        fid: string,
             feature identifier
        data: list of self.k arrays of shape(self.size[k], p) or self.size[k]
              the feature data
        override: bool, optional,
                  Allow feature overriding 
        """
        if len(data) != self.k:
            raise ValueError, 'data should have length k'
        for k in range(self.k):
            if data[k].shape[0] != self.size[k]:
                raise ValueError, 'Wrong data size'
        
        if (self.features.has_key(fid)) & (override==False):
            return
            
        self.features.update({fid:data})
            

    def get_feature(self, fid, k=None):
        """return self.features[fid]
        """
        if k==None:
            return self.features[fid]
        if k>self.k-1:
            raise ValueError, 'works only for k<%d'%self.k
        return self.features[fid][k]

    def representative_feature(self, fid, method='mean'):
        """
        Compute a statistical representative of the within-Foain feature

        Parameters
        ----------
        fid: string, feature id
        method: string, method used to compute a representative
                chosen among 'mean', 'max', 'median', 'min', 'weighted mean'
        """
        rf = []
        eps = 1.e-15
        for k in range(self.k):
            f = self.get_feature(fid, k)
            if method=="mean":
                rf.append(np.mean(f, 0))
            if method=="weighted mean":
                lvk = self.domain.local_volume[self.label==k]
                tmp = np.dot(lvk, f)/np.maximum(eps, np.sum(lvk))
                rf.append(tmp)
            if method=="min":
                rf.append( np.min(f, 0))
            if method=="max":
                rf.append(np.max(f, 0))
            if method=="median":
                rf.append(np.median(f, 0))
        return np.array(rf)
    
    def remove_feature(self, fid):
        """ Remove a certain feature
        """
        return self.features.pop(fid)
        

    def argmax_feature(self, fid):
        """ Return the list  of roi-level argmax of feature called fid
        """
        af = [np.argmax(self.feature[fid][k]) for k in range(self.k)]
        return np.array(af)


    def integrate(self, fid=None):
        """
        Integrate  certain feature on each ROI and return the k results

        Parameters
        ----------
        fid : string,  feature identifier,
              by default, the 1 function is integrataed, yielding ROI volumes

        Returns
        -------
        lsum = array of shape (self.k, self.feature[fid].shape[1]),
               the results
        """
        if fid==None:
            vol = [np.sum(self.domain.local_volume[self.label==k])
                   for k in range(self.k)] 
            return (np.array(vol))
        lsum = []
        for k in range(self.k):
            slvk = np.expand_dims(self.domain.local_volume[self.label==k], 1)
            sfk = self.features[fid][k]
            if sfk.size == sfk.shape[0]:
                sfk = np.reshape(sfk,(self.size[k], 1))
            sumk = np.sum(sfk*slvk, 0)
            lsum.append(sumk)
        return np.array(lsum)
        
    def check_features(self):
        """
        """
        pass

    def plot_feature(self, fid, ax=None):
        """
        boxplot the distribution of features within ROIs
        Note that this assumes 1-d features

        Parameters
        ----------
        fid: string,
             the feature identifier
        ax: axis handle, optional
        """
        f = self.get_feature(fid)
        #if f[0].shape[1]>1:
        #    raise ValueError, "cannot plot multi-dimensional\
        #    features for the moment"
        if ax is None:      
            import matplotlib.pylab as mp
            mp.figure()
            ax = mp.subplot(111)
        ax.boxplot(f)
        ax.set_title('ROI-level distribution for feature %s' %fid)
        ax.set_xlabel('Region index')
        ax.set_xticks(np.arange(1, self.k+1))#, np.arange(self.k))
        return ax
        
    def set_roi_feature(self, fid, data):
        """
        Parameters
        ----------
        fid: string, feature identifier
        data: array of shape(self.k, p), with p>0
        """
        if data.shape[0]!=self.k:
            print data.shape[0], self.k, fid
            raise ValueError, "Incompatible information of the provided data"
        if np.size(data)==self.k:
            data = np.reshape(data, (self.k, 1))
        self.roi_features.update({fid:data})

    def get_roi_feature(self, fid):
        """roi_features accessor
        """
        return self.roi_features[fid]

    def to_image(self, path=None):
        """
        Generates and possiblly writes a label image that represents self.

        Note
        ----
        Works only if self.dom is an ddom.NDGridDomain
        """
        if not isinstance(self.domain, ddom.NDGridDomain):
            return None

        tmp_image = self.domain.to_image()
        label = tmp_image.get_data().copy()-1
        label[label>-1] = self.label
        nim = Nifti1Image(label, tmp_image.get_affine())
        nim.get_header()['descrip'] = 'label image of %s' %self.id
        if path is not None:
            save(nim, path)
        return nim
    

def subdomain_from_array(labels, affine=None, nn=0):
    """
    return a SubDomain from an n-d int array
    
    Parameters
    ----------
    label: np.array instance
          a supposedly boolean array that yields the regions
    affine: np.array, optional
            affine transform that maps the array coordinates
            to some embedding space
            by default, this is np.eye(dim+1, dim+1)
    nn: int, neighboring system considered,
        unsued at the moment

    Note
    ----
    Only nonzero labels are considered
    """
    dom = ddom.grid_domain_from_array(np.ones(labels.shape), affine=affine,
                                      nn=nn)
    return SubDomains(dom, labels.astype(np.int))

def subdomain_from_image(mim, nn=18):
    """
    return a SubDomain instance from the input mask image

    Parameters
    ----------
    mim: NiftiIImage instance, or string path toward such an image
         supposedly a label image
    nn: int, optional
        neighboring system considered from the image
        can be 6, 18 or 26
        
    Returns
    -------
    The MultipleROI  instance

    Note
    ----
    Only nonzero labels are considered
    """
    if isinstance(mim, basestring):
        iim = load(mim)
    else :
        iim = mim

    return subdomain_from_array(iim.get_data(), iim.get_affine(), nn)

def subdomain_from_position_and_image(nim, pos):
    """
    keeps the set of labels of the image corresponding to a certain index
    so that their position is closest to the prescribed one
    Parameters
    ----------
    mim: NiftiIImage instance, or string path toward such an image
         supposedly a label image
    pos: array of shape(3) or list of length 3,
         the prescribed position
    """
    tmp = subdomain_from_image(nim)
    coord = np.array([tmp.domain.coord[tmp.label==k].mean(0)
                      for k in range(tmp.k)])
    idx = ((coord-pos)**2).sum(1).argmin()
    return subdomain_from_array(nim.get_data()==idx, nim.get_affine())
    
    
    
def subdomain_from_balls(domain, positions, radii):
    """
    Create discrete ROIs as a set of balls within a certain coordinate systems

    Parameters
    ----------
    domain: StructuredDomain instance,
            the description of a discrete domain
    positions: array of shape(k, dim):
               the positions of the balls
    radii: array of shape(k):
           the sphere radii
    """
    # checks
    if np.size(positions)==positions.shape[0]:
        positions = np.reshape(positions, (positions.size), 1)
    if positions.shape[1] !=  domain.em_dim:
        raise ValueError, 'incompatible dimensions for domain and positions'
    if positions.shape[0] != np.size(radii):
        raise ValueError, 'incompatible positions and radii provided'

    label = -np.ones(domain.size)

    for k in range(radii.size): 
        label[np.sum((domain.coord-positions[k])**2, 1)<radii[k]**2] = k

    return SubDomains(domain, label)

