# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from nipy.io.imageformats import load
from discrete_domain import\
     StructuredDomain, NDGridDomain, domain_from_array,\
     reduce_coo_matrix, array_affine_coord, smatrix_from_nd_array


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

    def plot_feature(self):
        """
        """
        pass

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
        """
        """
        return self.roi_features[fid]

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
    dom = domain_from_array(np.ones(labels.shape), affine=affine, nn=nn)
    return SubDomains(dom, labels.astype(np.int))

def subdomain_from_label_image(mim, nn=18):
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

#####################################################################
# MultipleROI
#####################################################################

class MultipleROI(object):
    """
    idem StructuredDomain (and DiscreteROI), but here coord,
    local_volume and topology are k-length lists of a arrays and
    coo-matrices
    """
    def __init__(self, dim, k, coord, local_volume, topology=None,
                 referential='', id='' ):
        """
        Parameters
        ----------
        dim: int,
             the (physical) dimension of the domain
        k: int,
           the number of ROIs consiedered in the model 
        coord: list of k arrays of shape(size[k], em_dim),
               explicit coordinates of the domain sites
        local_volume: list of k arrays of shape(size[k]),
                      yields the volume associated with each site
        topology: list of k sparse coo_matrices of shape (size[k], size[k]),
                  that yields the neighboring locations in the ROIs
        referential: string, optional,
                     identifier of the referential of the coordinates system
        id: string, optional,
                     identifier of the set of ROIs
                     
        Caveat
        ------
        em_dim may be greater than dim e.g. (meshes coordinate in 3D)

        Fixme:
        ------
        Local_volume and topology should be optional
        """
        # dimension
        self.dim = dim

        #k 
        self.k = int(np.maximum(k, 0))
        if len(coord) != self.k:
            raise ValueError, 'coord should have length %d' %self.k
        
        # number of discrete elements
        self.size = np.zeros(self.k, dtype=np.int)
        for k in range(self.k):
            self.size[k] = coord[k].shape[0]

        # coordinates
        for k in range(self.k):
            if np.size(coord[k]) == coord[k].shape[0]:
                coord[k] = np.reshape(coord[k], (np.size(coord[k]), 1))

        self.em_dim = coord[0].shape[1]
        for k in range(self.k):
            if coord[k].shape[1] != self.em_dim:
                raise ValueError, 'Inconsistant coordinate dimensions'        
        if self.em_dim<dim:
            raise ValueError, 'Embedding dimension cannot be smaller than dim'
        self.coord = []
        for k in range(self.k):
            self.coord.append(coord[k])

        # volume
        if len(local_volume) != self.k:
            raise ValueError, 'local_volume should have a length %d' %self.k
        self.local_volume = []
        for k in range(self.k):
            if np.size(local_volume[k])!= self.size[k]:
                raise ValueError, "Inconsistent Volume size"
            self.local_volume.append(np.ravel(local_volume[k]))
        
        # topology
        if topology is None:
            self.topology = None
        else:
            self.topology = []
            if len(topology) != self.k:
                raise ValueError, 'Topology should have length %d' %self.k
            for k in range(self.k):
                #if self.size[k]==1:
                #    continue
                if topology[k].shape != (self.size[k], self.size[k]):
                    raise ValueError, 'Incorrect shape for topological model'
                self.topology.append(topology[k])

        self.referential = referential
        self.features = {}

    def copy(self, id=''):
        """ Returns a copy of self
        """
        cp = MultipleROI( self.k, self.coord.copy(), self.local_volume.copy(),
                          referential=self.referential, id=id )
        for fid in self.features.keys():
            f = self.features.pop(fid)
            sf = [f[k].copy() for k in range(self.k)]
            cp.set_feature(fid, sf)
        return cp
        
    def get_coord(self, k):
        """ returns self.coord[k]
        """
        if k>self.k-1:
            raise ValueError, 'works only for k<%d'%self.k
        return self.coord[k]

    def get_size(self):
        """ returns size, k-length array
        """
        return self.size

    def get_volume(self, k):
        """ returns self.local_volume[k]
        """
        if k>self.k-1:
            raise ValueError, 'works only for k<%d'%self.k
        return self.local_volume[k]

    def select(self, valid, id='', auto=False):
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

        Returns
        -------
        the instance, id auto==False, nothing otherwise
        """
        if valid.size != self.k:
            raise ValueError, 'Invalid size for valid'

        svol = [self.local_volume[k] for k in range(self.k) if valid[k]]
        if self.topology is not None:
            stopo = [self.topology[k] for k in range(self.k) if valid[k]]
        else:
            stopo = None
        scoord = [self.coord[k] for k in range(self.k) if valid[k]]

        if auto:
            self.k = valid.sum()
            self.coord = scoord
            self.size = np.array([self.coord[k].shape[0]
                                  for k in range(self.k)])
            self.volume = svol
            self.topology = stopo
            for fid in self.features.keys():
                f = self.features.pop(fid)
                sf = [f[k] for k in range(self.k) if valid[k]]
                self.set_feature(fid, sf)
            
        else:
            DD = MultipleROI(self.dim, valid.sum(), scoord, svol, stopo,
                             self.referential, id)
        
            for fid in self.features.keys():
                f = self.features.pop(fid)
                sf = [f[k] for k in range(self.k) if valid[k]]
                DD.set_feature(fid, sf)

            return DD

    def set_feature(self, fid, data, override=True):
        """
        Append a feature 'fid'
        
        Parameters
        ----------
        fid: string,
             feature identifier
        data: list of self.k arrays of shape(self.size[k], p) or self.size[k]
              the feature data 
        """
        if len(data) != self.k:
            raise ValueError, 'data should have length k'
        for k in range(self.k):
            if data[k].shape[0]!=self.size[k]:
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
                chosen among 'mean', 'max', 'median', 'min' 
        """
        rf = []
        for k in range(self.k):
            f = self.get_feature(fid, k)
            if method=="mean":
                rf.append(np.mean(f, 0))
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
            vol = [np.sum(self.local_volume[k]) for k in range(self.k)] 
            return (np.array(vol))
        lsum = []
        for k in range(self.k):
            slvk = np.reshape(self.local_volume[k], (self.size[k], 1))
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

    def plot_feature(self):
        """
        """
        pass

def mroi_from_label_image(mim, nn=18):
    """
    return a MultipleROI instance from the input mask image

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

    return mroi_from_array(iim.get_data(), iim.get_affine(), nn)
    
def mroi_from_array(labels, affine=None, nn=0):
    """
    return a StructuredDomain from an n-d array
    
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
    dim = len(labels.shape)
    if affine is None:
        affine =  np.eye(dim + 1)
    ul = np.unique(labels[labels!=0])
    k = np.size(ul)
    vol = []
    coord = []
    topology = []
    vvol = np.absolute(np.linalg.det(affine))
    for q in ul:
        vol.append(vvol*np.ones(np.sum(labels==q)))
        coord.append(array_affine_coord(labels==q, affine))
        topology.append(smatrix_from_nd_array(labels==q, nn))
    return MultipleROI(dim, k, coord, vol, topology)
    

def mroi_from_balls(domain, positions, radii):
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
    coord = []
    vol = []
    topology = []
    for k in range(radii.size): 
        ik = np.sum((domain.coord-positions[k])**2, 1)<radii[k]**2
        coord.append(domain.coord[ik])
        vol.append(domain.local_volume[ik])
        topology.append(reduce_coo_matrix(domain.topology, ik))
    return MultipleROI(domain.dim, radii.size, coord, vol, topology)


class MultipleGridRoi(MultipleROI):

    def __init__(self, dim, k, idx, shape, affine, topology,
                 referential='', id=''  ):
        pass

    def check_header(self, image_path):
        """
        """
        pass
    
    def to_nifti1image(self, image_path=None):
        """
        """
        pass
    
    def to_array(self):
        """
        """
        pass
        
    def set_feature_from_image():
        """
        """
        pass

    

def mgrid_roi_from_label_image():
    """
    """
    pass

def mgrid_roi_from_balls():
    """
    """
    pass

