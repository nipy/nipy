# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from nipy.io.imageformats import load, save, Nifti1Image 
   

###############################################################################
# class DiscreteROI 
###############################################################################
        
    
class DiscreteROI(object):
    """
    Temporary ROI class for nipy
    Ultimately, it should be merged with the nipy class
    
    ROI definition requires
    an identifier
    an affine transform and a shape that can be used to
       translate grid coordinates to position and to 
       generate images from the DiscreteROI structure
    The ROI can be derived from a image 

    roi.features is a dictionary of informations on the ROI elements.
    It is assumed that the ROI is sampled on a discrete grid, so that
    each feature is in fact a (voxel,feature_dimension) array
    """

    def __init__(self, id="roi", affine=np.eye(4), shape=None):
        """
        roi = ROI(id='roi', header=None)

        Parameters
        -----------
        id: string
            roi identifier
        header: pynifty header 
            referential-defining information
        """
        self.id = id
        self.affine = affine
        self.shape = shape
        self.features = dict()

    def check_header(self, image_path):
        """
        checks that the image is in the header of self

        Parameters
        ----------
        image_path: (string) the path of an image (nifti)

        Returns
        -------
        True if the affine and shape of the given nifti file correpond
        to the ROI's.
        """
        eps = 1.e-15
        nim = load(image_path)
        if (np.absolute(nim.get_affine()-self.affine)).max() > eps:
            return False
        if self.shape is not None:
            return np.all(np.equal(nim.get_shape(), self.shape))
        return True

    def from_binary_image(self, image_path):
        """
        Take all the <> 0 sites of the image as the ROI

        Parameters
        -----------
        image_path: string
            the path of an image
        """
        self.check_header(image_path)
        nim = load(image_path)
        self.discrete = np.where(nim.get_data())
        
    def from_position(self, position, radius):
        """
        A ball in the grid
        requires that the grid and header are defined
        """
        if self.shape==None:
            raise ValueError, 'self.shape has to be defined'
        # define the positions associated with the grid
        grid = np.indices(self.shape)

        nvox = np.prod(self.shape)
        grid.shape = (3, nvox)
        grid = np.hstack((grid.T, np.ones((nvox, 1))))
        coord = np.dot(grid, self.affine.T)[:,:3]
        
        # finally derive the mask of the ROI
        dx = coord - position
        sqra = radius**2
        self.discrete = tuple(grid[np.sum(dx**2,1)<sqra,:3].T.astype(np.int))
        
    def from_labelled_image(self, image_path, label):
        """
        Define the ROI as the set of  voxels of the image
        that have the pre-defined label

        Parameters
        -----------
        image_path: ndarray
            a label (discrete valued) image
        label: int
            the desired label
        """
        self.check_header(image_path)
        nim = load(image_path)
        data = nim.get_data()
        self.discrete = np.where(data==label)
        
    def from_position_and_image(self, image_path, position):
        """
         Define the ROI as the set of  voxels of the image
         that is closest to the provided position

        Parameters
        -----------
        image_path: string, 
            the path of a label (discrete valued) image
        position: array of shape (3,)
            x, y, z position in the world space

        Notes
        -------
        everything could be performed in the image space
        """
        # check that the header is OK indeed
        self.check_header(image_path)

        # get the image data and find the best matching ROI
        nim = load(image_path)
        data = nim.get_data().astype(np.int)
        k = data.max()+1
        cent = np.array([np.mean(np.where(data==i),1) for i in range(k)])
        cent = np.hstack((cent,np.ones((k,1))))
        coord = np.dot(cent, self.affine.T)[:,:3]
        
        # find the best match
        dx = coord-position
        k = np.argmin(np.sum(dx**2,1))
        self.discrete = np.where(data==k)
        
    def make_image(self, image_path=None):
        """
        write a binary nifty image where the nonzero values are the ROI mask

        Parameters
        -----------
        image_path: string, optional 
            the desired image name
        """
        if self.shape==None:
            raise ValueError, 'self.shape has to be defined'
        data = np.zeros(self.shape)
        data[self.discrete] = 1
        
        wim = Nifti1Image(data, self.affine)
        wim.get_header()['descrip'] = "ROI image"
        if image_path !=None:
            save(wim, image_path)
        return wim

    def set_feature(self, fid, data):
        """
        Given an array of data that is assumed to comprise
        the ROI, get the subset of values that correponds to
        voxel-based data in the ROI
        
        Parameters
        -----------
        fid: string 
            feature identifier, e.g. data (array of shape (self.VolumeExtent))
        
        Returns
        --------
        ldata: ndarray of shape (roi.nbvox, dim)
            the ROI-based feature

        Notes
        ------
        this function creates a reduced feature array corresponding to the 
        ROI item.
        """
        if (np.shape(data)!=self.shape):
            raise ValueError, "incompatible dimension of provided data"
    
        ldata = data[self.discrete]
        self.features.update({fid:ldata})
        return ldata
        
    def set_feature_from_image(self, fid, image_path):
        """
        extract some roi-related information from an image

        Parameters
        -----------
        fid: string
            feature id
        image: string
            image path
        """
        self.check_header(image_path)
        nim = load(image_path)  
        data = nim.get_data()
        self.set_feature(fid,data)

    def get_feature(self, fid):
        """
        return the feature corrsponding to fid, if it exists
        """
        return self.features[fid]

    def set_feature_from_masked_data(self, fid, data, mask):
        """
        idem set_feature but the input data is thought to be masked
        """
        raise NotImplementedError
        
    def representative_feature(self, fid, method="mean"):
        """
        Compute a statistical representative of the within-ROI feature
        """
        f = self.get_feature(fid)
        if method=="mean":
            return np.mean(f,0)
        if method=="min":
            return np.min(f,0)
        if method=="max":
            return np.max(f,0)
        if method=="median":
            return np.median(f,0)
        
    def plot_feature(self, fid):
        """
        boxplot the feature within the ROI
        """
        f = self.get_feature(fid)
        import matplotlib.pylab as mp
        mp.figure()
        mp.boxplot(f)
        mp.title('Distribution of %s within %s'%(fid,self.id))
        

################################################################################
# class MultipleROI 
################################################################################

class MultipleROI(object):
    """
    This is  a class to deal with multiple discrete ROIs defined 
    in a given space
    (mroi.affine, mroi.shape) provide the necessary referential information
    so that the mroi is basically defined as a multiple 
    sets of 3D coordinates
    finally, there is two associated feature dictionaries:

    roi_features: list of roi-level features
                  it is assumed that each feature is an
                  (roi,feature_dim) array, 
                  i.e. each roi is assumed homogeneous
                  wrt the feature

    discrete_features is a dictionary of informations sampled
                      on the discrete membre of the rois 
                      (voxels or vertices)
                      each feature os thus a list of self.k arrays 
                      of shape (roi_size, feature_dimension)
    
    fixme :
    - can create an ROI with k =0 ? Deal with taht case properly
    - merge append_balls and as_multiple_balls
    """

    def __init__(self, id="multiple_roi", k=0, affine=np.eye(4), 
                       shape=None, xyz=None):
        """
        roi = MultipleROI(id='roi', header=None)

        Parameters
        ----------
        id="multiple_roi" string, roi identifier
        k=1, int, number of rois that are included in the structure 
        affine=np.eye(4), array of shape(4,4),
                           coordinate-defining affine transformation
        shape=None, tuple of length 3 defining the size of the grid 
                    implicit to the discrete ROI definition
        xyz=None: list of arrays of shape (nvox[i],3)
                  the grid coordinates of the rois elements
                          
        """
        self.id = id
        self.k = k
        self.affine = affine
        self.shape = shape
        if xyz==None:
            self.xyz = []
        else:
            self.xyz = xyz
        self.roi_features = dict()
        self.discrete_features = dict()
        self.check_consistency()

    def check_consistency(self):
        """
        Check the consistency of the input values:
        affine should be a (4,4) array
        
        all values of xyz should be in the range [0,d1]*[0,d2]*[0,d3]
        where self.shape = (d1,d2,d3), if shape is defined
     
        """
        # affine should be a (dim+1, dim+1) array
        dim = 3
        if self.shape is not None:
            dim = len(self.shape)
        if np.shape(self.affine)!= (dim+1, dim+1):
            raise ValueError, "affine does not have a correct shape"
       
        if (self.shape!=None)&(len(self.xyz)>0):
            xyzmin = np.min(np.array([np.min(self.xyz[k],0) 
                                     for k in range(self.k)]),0)
            xyzmax = np.max(np.array([np.max(self.xyz[k],0) 
                                     for k in range(self.k)]),0)
            if (xyzmin<0).any():
                raise ValueError, 'negative grid coordinates have been provided'
            if (xyzmax>np.array(self.shape)).any():
                raise ValueError, 'grid ccordinates are greater than\
                the provided shape'    

    def check_features(self):
        """
        check that self.roi_features have the coorect size
        i.e. f.shape[0]=self.k for f in self.roi_features
        and that self.discrete features have the correct size
        i.e.  for f in self.roi_features:
        f is a list of length self.k
        f[i] is an array with dimensions consistent with xyz

        Note: features that are not found consistent are removed
        """
        fids = self.roi_features.keys()
        for fid in fids:
            if self.roi_features[fid].shape[0]!=self.k:
                print "removing feature %s, which has incorrect size" %fid 
                self.roi_features.pop(fid)

        fids = self.discrete_features.keys()
        for fid in fids:
            dff = self.discrete_features[fid]
            if len(dff)!=self.k:
                print "removing feature %s, which has incorrect length" %fid 
                self.discrete_features.pop(fid)
            for k in range(self.k):
                if dff[k].shape[0]!=self.xyz[k].shape[0]:
                    print "removing feature %s, incorrectly shaped" %fid 
                    self.discrete_features.pop(fid)

    def check_header(self, image_path):
        """
        checks that the image is in the header of self

        Parameters
        ----------
        image_path: (string) the path of an image
        """
        eps = 1.e-15
        nim = load(image_path)
        b = True
        if (np.absolute(nim.get_affine()-self.affine)).max()>eps:
            b = False
        if self.shape!=None:
            for d1,d2 in zip(nim.get_shape(),self.shape):
                if d1!=d2:
                    b = False
        return b

    def from_labelled_image(self, image_path, labels=None, add=True):
        """
        All the voxels of the image that have non-zero-value
        self.k becomes the number of values of the (discrete) image

        Parameters
        ----------
        image_path: string
            path of a label (discrete valued) image
        labels=None : array of shape (nlabels) 
                    the set of image labels that
                    shall be used as ROI definitions
                    By default, all the image labels are used
        
        Note
        ----
        this can be used to append roi_features,
        when rois are already defined
        """
        self.check_header(image_path)
        nim = load(image_path)
        data = nim.get_data()
        udata = np.unique(data[data>0])
        if labels==None:    
            if add: self.k += np.size(udata)
            for k in range(np.size(udata)):
                dk = np.array(np.where(data==udata[k])).T
                self.xyz.append(dk)
        else:
            if add: self.k += np.size(labels)
            for k in range(np.size(labels)):
                if (data==labels[k]).any():
                    dk = np.array(np.where(data==udata[k])).T
                    self.xyz.append(dk)
                else:
                    raise ValueError, "Sorry I don't take empty ROIs"
        self.check_features()

    def as_multiple_balls(self, position, radius):
        """
        self.as_multiple_balls(position, radius)
        Given a set of positions and radii, defines one roi
        at each (position/radius) couple 

        Parameters
        ----------
        position: array of shape (k,3): the set of positions
        radius: array of shape (k): the set of radii
        """
        if self.shape==None:
            raise ValueError, "Need self.shape to be defined for this function"
        
        if np.shape(position)[0]!=np.size(radius):
            raise ValueError, "inconsistent position/radius definition"

        if np.shape(position)[1]!=3:
            raise ValueError, "This function takes only 3D coordinates:\
            provide the positions as a (k,3) array"

        self.k = np.size(radius)
        
        # define the positions associated with the grid
        grid = np.indices(self.shape)
        nvox = np.prod(self.shape)
        grid.shape = (3, nvox)
        grid = np.hstack((grid.T, np.ones((nvox, 1))))
        coord = np.dot(grid, self.affine.T)[:,:3]
        
        # finally derive the mask of the ROI
        for k in range(self.k):
            dx = coord - position[k]
            sqra = radius[k]**2
            dk = grid[np.sum(dx**2,1)<sqra,:3].astype(np.int)
            self.xyz.append(dk)
        self.set_roi_feature('position',position)
        self.set_roi_feature('radius',radius)
        self.check_features()


    def append_balls(self, position, radius):
        """
        idem self.as_multiple_balls, but the ROIs are added
        fixme : should be removed from the class
        as soon as __add__ is implemented
        """
        if self.shape==None:
            raise ValueError, "Need self.shape to be defined for this function"
        
        if np.shape(position)[0]!=np.size(radius):
            raise ValueError, "inconsistent position/radius definition"

        if np.shape(position)[1]!=3:
            raise ValueError, "Sorry, I take only 3D regions: \
            provide the positions as a (k,3) array"

        self.k += np.size(radius)
        
        # define the positions associated with the grid
        grid = np.indices(self.shape)
        nvox = np.prod(self.shape)
        grid.shape = (3, nvox)
        grid = np.hstack((grid.T, np.ones((nvox, 1))))
        coord = np.dot(grid, self.affine.T)[:,:3]
        grid = grid[:,:3]
        
        # finally derive the mask of the ROI
        for k in range(np.size(radius)):
            dx = coord - position[k]
            sqra = radius[k]**2
            dk = grid[np.sum(dx**2,1)<sqra,:3].astype(np.int)
            self.xyz.append(dk)

        if self.roi_features.has_key('position'):
            self.complete_roi_feature('position',position)
        if self.roi_features.has_key('radius'):
            self.complete_roi_feature('radius',radius)

        self.check_features()

    def complete_roi_feature(self, fid, values):
        """
        completes roi_feature corresponding to fid
        by appending the values
        """
        f = self.roi_features.pop(fid)
        if values.shape[0]==np.size(values):
            values = np.reshape(values,(np.size(values),1))
        if f.shape[1]!=values.shape[1]:
            raise ValueError, "incompatible dimensions for the roi_features"
        f = np.vstack((f,values))
        self.set_roi_feature(fid,f)
        
        
    def make_image(self, path=None):
        """
        write a int image where the nonzero values are the ROIs

        Parameters
        ----------
        path: string, optional
            the desired image path

        Returns
        -------
        brifti image instance
        
        Note
        ----
        the background values are set to -1
        the ROIs values are set as [0..self.k-1]
        """
        if self.shape==None:
            raise ValueError, 'Need self.shape to be defined'

        data = -np.ones(self.shape,np.int)
        for k in range(self.k):
            dk = self.xyz[k].T
            data[dk[0], dk[1], dk[2]] = k

        wim =  Nifti1Image(data, self.affine)
        header = wim.get_header()
        header['descrip'] = "Multiple ROI image"
        if path!=None:
            save(wim, path)
        return wim
  
    def set_roi_feature(self,fid,data):
        """
        this function simply stores data 

        Parameters
        ----------
        fid (string): feature identifier, e.g.
        data: array of shape(self.k,p),with p>0
        """
        if data.shape[0]!=self.k:
            print data.shape[0], self.k, fid
            raise ValueError, "Incompatible information of the provided data"
        if np.size(data)==self.k:
            data = np.reshape(data,(self.k,1))
        self.roi_features.update({fid:data})

    def set_discrete_feature(self,fid,data):
        """
        Parameters
        ----------
        fid (string): feature identifier
        data: list of self.k arrays with shape(nk,p),with p>0
              nk = self.xyz[k].shape[0] (number of elements in ROI k)
              this function simply stores data 
        """
        if len(data)!=self.k:
            print len(data), self.k, fid
            raise ValueError, "Incompatible information of the provided data"
        for k in range(self.k):
            datak = data[k]
            if datak.shape[0]!=self.xyz[k].shape[0]:
                raise ValueError, "badly shaped data"
            if np.size(datak)==self.xyz[k].shape[0]:
                datak = np.reshape(datak,(np.size(datak),1))
        self.discrete_features.update({fid:data})

    def set_roi_feature_from_image(self, fid, image_path, method='average'):
        """
        extract some roi-related information from an image
        
        Parameters
        ----------
        fid: feature id
        image_path(string): path of the feature-defining image
        method='average' (string) : take the roi feature as
                         the average feature over the ROI
       
        """
        self.check_header(image_path)
        nim = load(image_path)  
        data = nim.get_data()
        ldata = np.zeros((self.k,1))
        for k in range(self.k):
            dk = self.xyz[k].T
            ldata[k] = np.mean(data[dk[0],dk[1],dk[2]])
        self.set_roi_feature(fid,ldata)

    def set_discrete_feature_from_image(self, fid, image_path=None,
                                        image=None):
        """
        extract some discrete information from an image

        Parameters
        ----------
        fid: string, feature id
        image_path, string, optional
            input image path
        image, brfiti image path,
            input image

        Note that either image_path or image has to be provided
        """
        if image_path==None and image==None:
            raise ValueError, "one image needs to be provided"
        if image_path is not None:
            self.check_header(image_path)
            nim = load(image_path)
        if image is not None:
            nim = image
        data = nim.get_data()
        ldata = []
        for k in range(self.k):
            dk = self.xyz[k].T
            ldk = data[dk[0],dk[1],dk[2]]
            if np.size(ldk)==ldk.shape[0]:
                ldk = np.reshape(ldk,(np.size(ldk),1))
            ldata.append(ldk)
        self.set_discrete_feature(fid,ldata)

    def set_discrete_feature_from_index(self, fid, data):
        """
        Assuming that self.discrete_feature['index'] exists
        this extracts the values from data corresponding to the index
        and sets these are self.discrete_feature[fid]

        Parameters
        ----------
        fid (string): feature id
        data: array of shape(nbitem,k) where nbitem is supposed
              to be greater than any value in 
              self.discrete_feature['index']

        Note
        ----
        This function implies that the users understand what they do
        In particular that they know what  self.discrete_feature['index']
        corresponds to.
        """
        if self.discrete_features.has_key('index')==False:
            raise ValueError, 'index has not been defined as a\
            discrete feature'
        index = self.discrete_features['index']
        imax = np.array([i.max() for i in index]).max()
        if imax>data.shape[0]:
            raise ValueError,\
                  'some indices are greater than input data size'
        ldata = []
        for k in range(self.k):
            ldata.append(data[np.ravel(index[k])])
        self.set_discrete_feature(fid,ldata)

    def discrete_to_roi_features(self, fid, method='average'):
        """
        Compute an ROI-level feature given the discrete features

        Parameters
        ----------
        fid(string) the discrete feature under consideration
        method='average' the assessment method

        Returns
        -------
        ldata: array of shape [self.k,fdim ]
               the computed roi-level feature 
        """
        data = self.discrete_features[fid]
        d0 = data[0]
        if np.size(d0) == np.shape(d0)[0]:
            d0 = np.reshape(d0,(np.size(d0),1))
        fdim = d0.shape[1]
        ldata = np.zeros((self.k,fdim))
        for k in range(self.k):
            if method=='average':
                ldata[k] = np.mean(data[k],0)
            if method == 'min':
                ldata[k] = np.min(data[k],0)
            if method == 'max':
                ldata[k] = np.max(data[k],0)
            if method not in['min','max','average']:
                print 'not implemented yet'
        self.set_roi_feature(fid,ldata)   
        return ldata

    def get_roi_feature(self,fid):
        """return sthe searched feature
        """
        return self.roi_features[fid]

    def remove_roi_feature(self,fid):
        """removes the specified feature
        """
        self.roi_features.pop(fid)
    
    def feature_argmax(self,fid):
        """
        Returns for each roi the index of the discrete element
        that is the within-ROI for the fid feature
        this makes sense only if the corresponding feature 
        has dimension 1
        """
        df = self.discrete_features[fid]
        if np.size(df[0])>np.shape(df[0])[0]:
            print "multidimensional feature; argmax is ambiguous"
        idx = -np.ones(self.k).astype(np.int)
        for k in range(self.k):
            idx[k] = np.argmax(df[k])
        return idx
            
    def plot_roi_feature(self, fid):
        """
        boxplot the feature within the ROI
        Note that this assumes a 1-d feature

        Parameters
        ----------
        fid string,
            the feature identifier
        """
        f = self.roi_features[fid]
        if f.shape[1]>1:
            raise ValueError, "cannot plot multi-dimensional\
            features for the moment"
        import matplotlib.pylab as mp
        ax = mp.figure()
        mp.bar(np.arange(self.k)+0.5,f)
        mp.title('ROI-level value for feature %s' %fid)
        mp.xlabel('ROI index')
        mp.xticks(np.arange(1, self.k+1),np.arange(1, self.k+1))
        return ax

    def plot_discrete_feature(self, fid, ax=None):
        """
        boxplot the distribution of features within ROIs
        Note that this assumes 1-d features

        Parameters
        ----------
        fid: string,
             the feature identifier
        ax: axis handle, optional
        """
        f = self.discrete_features[fid]
        if f[0].shape[1]>1:
            raise ValueError, "cannot plot multi-dimensional\
            features for the moment"
        if ax is None:      
            import matplotlib.pylab as mp
            ax = mp.figure()
        ax.boxplot(f)
        ax.set_title('ROI-level distribution for feature %s' %fid)
        ax.set_xlabel('ROI index')
        ax.set_xticks(np.arange(1, self.k+1))#np.arange(1, self.k+1))
        return ax


    def clean(self,valid):
        """
        remove the regions for which valid==0

        Parameters
        ----------
        valid: (boolean) array of shape self.k
        """
        if np.size(valid)!=self.k:
            raise ValueError, "the valid marker does not have\
            the correct size"

        self.xyz = [self.xyz[k] for k in range(self.k) if valid[k]]
        kold = self.k
        self.k = np.sum(valid.astype(np.int))
        
        for fid in self.roi_features.keys():
            f = self.roi_features.pop(fid)
            f = f[valid]
            self.set_roi_feature(fid,f)

        for fid in self.discrete_features.keys():
            f = self.discrete_features.pop(fid)
            nf = [f[k] for k in range(kold) if valid[k]]
            self.set_discrete_feature(fid, nf)

        self.check_features()

    def get_size(self):
        """
        return the number of voxels per ROI in one array
        """
        size = np.zeros(self.k).astype(np.int)
        for k in range(self.k):
            size[k] = np.shape(self.xyz[k])[0]
        return size

    def set_xyz(self, xyz):
        """
        set manually the values of xyz
        xyz is a list of arrays that contains
        the coordinates of all ROIs voxels

        Parameters:
        -----------
        xyz: list of length k containing inedx/coordinate arrays,
        one for each ROI
        """
        if len(xyz)!= self.k:
            raise ValueError, "the provided values for xyz \
            do not match self.k" 
        self.xyz = xyz

    def compute_discrete_position(self):
        """
        Create a 'position' feature based on self.affine
        and self.indexes, which is simply an affine transform
        from self.xyz to the space of self

        fixme : if a position is already available it does not
        need to be computed

        the computed position is returned
        """
        
        pos = []
        for  k in range(self.k):
            grid = self.xyz[k]
            nvox = grid.shape[0]
            grid = np.hstack((grid, np.ones((nvox, 1))))
            coord = np.dot(grid, self.affine.T)[:,:3]
            pos.append(coord)

        self.set_discrete_feature('position',pos)   
        return pos

    def append_discrete_ROI(self, droi):
        """
        complete self with a discrete roi
        only the features that have a common ideas between self and droi
        are kept
        """
        print 'Not implemented yet'
        pass
