import numpy as np
import nifti

################################################################################
# class `ROI`
################################################################################
class ROI(object):
    """
    Temporary ROI class for fff
    Ultimately, it should be merged with the nipy class
    
    ROI definition requires
    - an identifier
    - an header (exactly a nifti header at the moment,
    though not everything is necessary)
    The ROI can be derived from a image or defined
    in the coordinate system implied by header.sform()

    roi.features is a dictionary of informations on the ROI elements.
    It is assumed that the ROI is sampled on a discrete grid, so that
    each feature is in fact a (voxel,feature_dimension) array
    """

    def __init__(self, id="roi", header=None):
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
        self.header = header
        self.features = dict()

    def check_header(self, image):
        """
        Checks that the image is in the header of self

        Parameters
        -----------
        image: string
            the path of an image
        """
        #print "check not implemented yet"
        eps = 1.e-15
        nim = nifti.NiftiImage(image)
        header  = nim.header
        b = True
        if (np.absolute(header['sform']-self.header['sform'])).max()>eps:
            b = False
        for d1,d2 in zip(header['dim'],self.header['dim']):
            if d1!=d2:
                b = False
        return b

    def from_binary_image(self, image):
        """
        Take all the <>0 sites of the image as the ROI

        Parameters
        -----------
        image: string
            the path of an image
        """
        self.check_header(image)
        nim = nifti.NiftiImage(image)
        data = nim.data.T
        self.discrete = np.where(data)
        
    def from_position(self, position, radius):
        """
        A ball in the grid
        requires that the grid and header are defined
        """
        # check that the ref is defined
        if self.header==None:
            raise ValueError, "No defined referntial"

        # define the positions associated with the grid
        sform = self.header['sform']
        dim = self.header['dim'][1:4]
        grid = np.indices(dim)

        nvox = np.prod(dim)
        grid.shape = (3, nvox)
        grid = np.hstack((grid.T, np.ones((nvox, 1))))
        coord = np.dot(grid, sform.T)[:,:3]
        
        # finally derive the mask of the ROI
        dx = coord - position
        sqra = radius**2
        self.discrete = tuple(grid[np.sum(dx**2,1)<sqra,:3].T.astype(np.int))
        

    def from_labelled_image(self,image,label):
        """
        Define the ROI as the set of  voxels of the image
        that have the pre-defined label

        Parameters
        -----------
        image: ndarray
            a nifti label (discrete valued) image
        label: int
            the desired label
        """
        self.check_header(image)
        nim = nifti.NiftiImage(image)
        data = nim.data.T
        self.discrete = np.where(data==label)
        
        
    def from_position_and_image(self,image,position):
        """
         Define the ROI as the set of  voxels of the image
         that is closest to the provided position

        Parameters
        -----------
        image: string, 
            the path of a nifti label (discrete valued) image
        position: array of shape (3,)
            x, y, z position in the world space

        Notes
        -------
        everything could be performed in the image space
        """
        # check that the header is OK indeed
        self.check_header(image)
        
        # get the coordinates of the regions
        sform = self.header['sform']
        nim = nifti.NiftiImage(image)
        data = nim.data.T.astype(np.int)
        k = data.max()+1
        cent = np.array([np.mean(np.where(data==i),1) for i in range(k)])
        cent = np.hstack((cent,np.ones((k,1))))
        coord = np.dot(cent,np.transpose(sform))[:,:3]
        
        # find the best match
        dx = coord-position
        k = np.argmin(np.sum(dx**2,1))
        self.discrete = np.where(data==k)
        
    def make_image(self,name):
        """
        write a binary nifty image where the nonzero values are the ROI mask

        Parameters
        -----------
        name: string 
            the desired image name
        """
        #if name==None: name = "%s.nii" % self.id
        data = np.zeros(tuple(self.header['dim'][1:4]))
        data[self.discrete]=1
        data = np.reshape(data,tuple(self.header['dim'][1:4])).T
        nim = nifti.NiftiImage(data,self.header)
        nim.description = "ROI image"
        nim.save(name)

    def set_feature(self,fid,data):
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
        IVE = np.shape(data)
        VE = tuple(self.header['dim'][1:4])
        if IVE!= VE:
            raise ValueError, "incompatible dimension of provided data"
    
        ldata = data[self.discrete]
        self.features.update({fid:ldata})
        return ldata
        
    def set_feature_from_image(self,fid,image):
        """
        extract some roi-related information from an image

        Parameters
        -----------
        fid: string
            feature id
        image: string
            image path
        """
        nim = nifti.NiftiImage(image)  
        header = nim.header
        data = nim.asarray().T
        self.set_feature(fid,data)

    def get_feature(self,fid):
        """
        return the feature corrsponding to fid, if it exists
        """
        return self.features[fid]

    def set_feature_from_masked_data(self,fid,data,mask):
        """
        idem set_feature but the input data is thought to be masked
        """
        pass
        
    def representative_feature(self,fid,method="mean"):
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
        
    def plot_feature(self,fid):
        """
        boxplot the feature within the ROI
        """
        f = self.get_feature(fid)
        import matplotlib.pylab as mp
        mp.figure()
        mp.boxplot(f)
        mp.title('Distribution of %s within %s',%(fid,self.id))
        


################################################################################
# class `WeightedROI`
################################################################################
class WeightedROI(ROI):
    """
    ROI where a weighting is defined on the voxels
    """
    
    def __init__(self,id="roi",header=None,grid = None):
        """
        """
        
        pass


################################################################################
# class `MultipleROI`
################################################################################
class MultipleROI(object):
    """
    This is  a class to deal with multiple ROIs defined in a given space
    mroi.header is assumed to provide all the referential information
    (this should be changed in the future),
    so that the mroi is basically defined as a multiple sets of 3D coordinates
    finally, there is two associated feature dictionaries:

    roi.roi_features: roi-level features
    it is assumed that each feature is an
    (roi,feature_dim) array, i.e. each roi is assumed homogeneous
    wrt the feature
    
    roi.discrete_features is a dictionary of informations sampled
    on the discrete membre of the rois (voxels or vertices)
    each feature os thus a list of self.k arrays of shape
    (roi_size, feature_simension)

    """

    def __init__(self, id="roi", k=0,header=None,discrete=None):
        """
        roi = MultipleROI(id='roi', header=None)
        - id (string): roi identifier
        - k: number of rois that are included in the structure 
        - header (nipy header) : referential-defining information
        - discrete=None: list of index arrays
        that represent the grid coordinates of the rois elements
        """
        self.id = id
        self.k = k
        self.header = header
        self.discrete = discrete
        self.roi_features = dict()
        self.discrete_features = dict()

    def check_features(self):
        """
        check that self.roi_features have the coorect size
        i.e. f.shape[0]=self.k for f in self.roi_features
        and that self.discrete features have the correct size
        i.e.  for f in self.roi_features:
        - f is a list of length self.k
        - f[i] is an array with dimensions consistent with discrete

        caveat: features that are not found consistent are removed
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
                if dff[k].shape[0]!=self.discrete[k].shape[0]:
                    print "removing feature %s, incorrectly shaped" %fid 
                    self.discrete_features.pop(fid)

    def check_header(self, image):
        """
        checks that the image is in the header of self
        INPUT:
        - image: (string) the path of an image
        """
        #print "check not implemented yet"
        eps = 1.e-15
        nim = nifti.NiftiImage(image)
        header = nim.header
        b = True

        if (np.absolute(header['sform']-self.header['sform'])).max()>eps:
            b = False
        for d1,d2 in zip(header['dim'],self.header['dim']):
            if d1!=d2:
                b = False
        return b

    def from_labelled_image(self,image,labels=None,add=True):
        """
        All the voxels of the image that have non-zero-value
        self.k becomes the number of values of the (discrete) image
        INPUT:
        - image (string): a nifti label (discrete valued) image
        -labels=None : the set of image labels that
        shall be used as ROI definitions
        By default, all the image labels are used
        note that this can be used to append roi_features,
        when rois are already defined
        """
        self.check_header(image)
        nim = nifti.NiftiImage(image)
        data = nim.data.T
        udata = np.unique(data[data>0])
        if labels==None:    
            if add: self.k+=np.size(udata)
            for k in range(np.size(udata)):
                dk = np.array(np.where(data==udata[k])).T
                self.discrete.append(dk)
        else:
            if add: self.k+=np.size(labels)
            for k in range(np.size(labels)):
                if np.sum(data==labels[k])>0:
                    dk = np.array(np.where(data==udata[k])).T
                    self.discrete.append(dk)
                else:
                    raise ValueError, "Sorry I don't take empty ROIs"
        self.check_features()

    def as_multiple_balls(self, position, radius):
        """
        self.as_multiple_balls(position, radius)
        Given a set of positions and radii, defines one roi
        at each (position/radius) couple 
        INPUT:
        position: array of shape (k,3): the set of positions
        radius: array of shape (k): the set of radii
        """
        # check that the ref is defined
        if self.header==None:
            raise ValueError, "No defined referential"

        if np.shape(position)[0]!=np.size(radius):
            raise ValueError, "inconsistent position/radius definition"

        if np.shape(position)[1]!=3:
            raise ValueError, "Sorry, I take only 3D regions: provide the positions as a (k,3) array"

        self.k=np.size(radius)
        
        # define the positions associated with the grid
        sform = self.header['sform']
        dim = self.header['dim'][1:4]
        grid = np.indices(dim)

        nvox = np.prod(dim)
        grid.shape = (3, nvox)
        grid = np.hstack((grid.T, np.ones((nvox, 1))))
        coord = np.dot(grid, sform.T)[:,:3]
        
        # finally derive the mask of the ROI
        for k in range(self.k):
            dx = coord - position[k]
            sqra = radius[k]**2
            dk = grid[np.sum(dx**2,1)<sqra,:3].astype(np.int)
            self.discrete.append(dk)
        self.set_roi_feature('position',position)
        self.set_roi_feature('radius',radius)
        self.check_features()


    def append_balls(self,position, radius):
        """
        idem self.as_multiple_balls, but the ROIs are added
        """
        if self.header==None:
            raise ValueError, "No defined referential"

        if np.shape(position)[0]!=np.size(radius):
            raise ValueError, "inconsistent position/radius definition"

        if np.shape(position)[1]!=3:
            raise ValueError, "Sorry, I take only 3D regions: provide the positions as a (k,3) array"

        self.k += np.size(radius)
        
        # define the positions associated with the grid
        sform = self.header['sform']
        dim = self.header['dim'][1:4]
        grid = np.indices(dim)

        nvox = np.prod(dim)
        grid.shape = (3, nvox)
        grid = np.hstack((grid.T, np.ones((nvox, 1))))
        coord = np.dot(grid, sform.T)[:,:3]
        
        # finally derive the mask of the ROI
        for k in range(np.size(radius)):
            dx = coord - position[k]
            sqra = radius[k]**2
            dk = grid[np.sum(dx**2,1)<sqra,:3].astype(np.int)
            self.discrete.append(dk)

        if self.roi_features.has_key('position'):
            self.complete_roi_feature('position',position)
        if self.roi_features.has_key('radius'):
            self.complete_roi_feature('radius',radius)

        self.check_features()

    def complete_roi_feature(self,fid,values):
        """
        completes roi_feature by appending the values
        """
        f = self.roi_features.pop(fid)
        if values.shape[0]==np.size(values):
            values = np.reshape(values,(np.size(values),1))
        if f.shape[1]!=values.shape[1]:
            raise ValueError, "incompatible dimensions for the roi_features"
        f = np.vstack((f,values))
        self.set_roi_feature(fid,f)
        
        
    def make_image(self,name):
        """
        write a int nifti image where the nonzero values are the ROIs
        INPUT:
        - the desired image name
        NOTE:
        - the background values are set to -1
        - the ROIs values are set as [0..self.k-1]
        """
        data = -np.ones(tuple(self.header['dim'][1:4]),'i')
        for k in range(self.k):
            dk = self.discrete[k].T
            data[dk[0],dk[1],dk[2]]=k
        data = np.reshape(data,tuple(self.header['dim'][1:4])).T
        nim = nifti.NiftiImage(data,self.header)
        nim.description = "ROI image"
        nim.save(name)
  
    def set_roi_feature(self,fid,data):
        """
        INPUT:
        - fid (string): feature identifier, e.g.
        - data: array of shape(self.k,p),with p>0
        this function simply stores data 
        """
        if data.shape[0]!=self.k:
            print data.shape[0],self.k,fid
            raise ValueError, "Incompatible information of the provided data"
        if np.size(data)==self.k:
            data = np.reshape(data,(self.k,1))
        self.roi_features.update({fid:data})

    def set_discrete_feature(self,fid,data):
        """
        INPUT:
        - fid (string): feature identifier, e.g.
        - data: list of self.k arrays with shape(nk,p),with p>0
        nk = self.discrete[k].shape[0] (number of elements in ROI k)
        this function simply stores data 
        """
        if len(data)!=self.k:
            print len(data),self.k,fid
            raise ValueError, "Incompatible information of the provided data"
        for k in range(self.k):
            datak = data[k]
            if datak.shape[0]!=self.discrete[k].shape[0]:
                raise ValueError, "badly shaped data"
            if np.size(datak)==self.discrete[k].shape[0]:
                datak = np.reshape(datak,(np.size(datak),1))
        self.discrete_features.update({fid:data})

    def set_roi_feature_from_image(self,fid,image,method='average'):
        """
        extract some roi-related information from an image
        INPUT:
        - fid: feature id
        - image(string): image name
        - method='average' (string) : take the roi feature as
        the average feature over the ROI
        CAVEAT : deprecated
        use set_discrete_feature_from_image()
        and discrete_to_roi_fetaures() instead
        """
        self.check_header(image)
        nim = nifti.NiftiImage(image)  
        header = nim.header
        data = nim.asarray().T
        ldata = np.zeros((self.k,1))
        for k in range(self.k):
            dk = self.discrete[k].T
            ldata[k] = np.mean(data[dk[0],dk[1],dk[2]])
        self.set_roi_feature(fid,ldata)

    def set_discrete_feature_from_image(self,fid,image):
        """
        extract some discrete information from an image
        INPUT:
        - fid: feature id
        - image(string): image name
        """
        self.check_header(image)
        nim = nifti.NiftiImage(image)  
        header = nim.header
        data = nim.asarray().T
        ldata = []
        for k in range(self.k):
            ldata.append(data[self.discrete[k]])
        self.set_discrete_feature(fid,ldata)    

    def discrete_to_roi_features(self,fid,method='average'):
        """
        Compute an ROI-level feature given the discrete features
        INPUT:
        - fid(string) the discrete feature under consideration
        - method='average' the assessment method
        OUPUT:
        the computed roi-feature is returned
        """
        df = self.discrete_features[fid]
        data = self.discrete_features[fid]
        d0 = data[0]
        if np.size(d0) == np.shape(d0)[0]:
            np.reshape(d0,(np.size(d0),1))
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
        """return sthe serached feature
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
        this makes sense only if the corresponding feature has dimension 1
        """
        df = self.discrete_features[fid]
        if np.size(df[0])>np.shape(df[0])[0]:
            print "multidimensional feature; argmax is ambiguous"
        idx = -np.ones(self.k).astype(np.int)
        for k in range(self.k):
            idx[k] = np.argmax(df[k])
        return idx
            
    def plot_roi_feature(self,fid):
        """
        boxplot the feature within the ROI
        Note that this assumes a 1-d feature
        INPUT:
        - fid the feature identifier
        """
        f = self.roi_features[fid]
        if f.shape[1]>1:
            raise ValueError, "cannot plot multi-dimensional feature for the moment"
        import matplotlib.pylab as mp
        mp.figure()
        mp.bar(np.arange(self.k),f)
        mp.show()

    def clean(self,valid):
        """
        remove the regions for which valid<=0
        """
        if np.size(valid)!=self.k:
            raise ValueError, "the valid marker does not have the correct size"

        self.discrete = [self.discrete[k] for k in range(self.k) if valid[k]]
        kold = self.k
        self.k = np.sum(valid.astype(np.int))
        
        for fid in self.roi_features.keys():
            f = self.roi_features.pop(fid)
            f = f[valid]
            self.set_roi_feature(fid,f)

        for fid in self.discrete_features.keys():
            f = self.discrete_features.pop(fid)
            nf = [f[k] for k in range(kold) if valid[k]]
            self.set_discrete_feature(fid,nf)

        self.check_features()

    def get_size(self):
        """
        return the number of voxels per ROI in one array
        """
        size = np.zeros(self.k).astype(np.int)
        for k in range(self.k):
            size[k] = np.shape(self.discrete[k])[0]
        return size

    def set_discrete(self,discrete):
        """
        set manually the values of discrete
        discrete is a list of arrays that contains
        the coordinates of all ROIs voxels
        len(discrete) must be equal to self.k
        """
        if len(discrete)!= self.k:
            raise ValueError, "the provided values for discrete \
            do not match self.k" 
        self.discrete = discrete

    def compute_discrete_position(self):
        """
        Create a 'position' feature based on self.header
        and self.indexes, which is simply an affine transform
        from self.discrete to the space of self
        it is assumed that self.header has a sform
        if not, the sform is assumed to be th identity

        fixme : if a position is already available it does not
        need to be computed

        the computed position is returned
        """
        bproblem = 1
        if isinstance(self.header,dict):
            if self.header.has_key('sform'):
                sform = self.header['sform']
                bproblem=0
                
        if bproblem:
            print "warning: no sform found for position definition, ",
            print "assuming it is the identity"
            sform = np.eye(4)

        pos = []
        for  k in range(self.k):
            grid = self.discrete[k]
            nvox = grid.shape[0]
            grid = np.hstack((grid, np.ones((nvox, 1))))
            coord = np.dot(grid, sform.T)[:,:3]
            pos.append(coord)

        self.set_discrete_feature('position',pos)   
        return pos
        

# XXX: We need to use test data shipped with nipy to do that.
def test1(verbose = 0):
    nim = nifti.NiftiImage("/tmp/spmT_0024.nii")
    header = nim.header
    dat = nim.asarray().T
    roi = ROI("myroi",header)
    roi.from_position(np.array([0,0,0]),5.0)
    roi.make_image("/tmp/myroi.nii")
    roi.set_feature_from_image('activ',"/tmp/spmT_0024.img")
    if verbose: roi.plot_feature('activ')
    return roi

def test2(verbose=0):
    nim = nifti.NiftiImage("/tmp/spmT_0024.nii")
    header = nim.header
    roi = ROI(header=header)
    roi.from_labelled_image("/tmp/blob.nii",1)
    roi.make_image("/tmp/roi2.nii")
    
    roi.from_position_and_image("/tmp/blob.nii",np.array([0,0,0]))
    roi.make_image("/tmp/roi3.nii")

    roi.set_feature_from_image('activ',"/tmp/spmT_0024.nii")
    mactiv = roi.representative_feature('activ')
    if verbose: roi.plot_feature('activ')
    return roi


def test_mroi1(verbose=0):
    nim =  nifti.NiftiImage("/tmp/blob.nii")
    header = nim.header
    mroi = MultipleROI(header=header)
    mroi.from_labelled_image("/tmp/blob.nii")
    mroi.make_image("/tmp/mroi.nii")
    mroi.set_roi_feature_from_image('activ',"/tmp/spmT_0024.nii")
    if verbose: mroi.plot_feature('activ')
    return mroi

def test_mroi2(verbose=0):
    nim =  nifti.NiftiImage("/tmp/blob.nii")
    header = nim.header
    mroi = MultipleROI(header=header)
    pos = 1.0*np.array([[10,10,10],[0,0,0],[20,0,20],[0,0,35]])
    rad = np.array([5.,6.,7.,8.0])
    mroi.as_multiple_balls(pos,rad)
    mroi.append_balls(np.array([[-10.,0.,10.]]),np.array([7.0]))
    mroi.make_image("/tmp/mroi.nii")
    mroi.set_roi_feature_from_image('activ',"/tmp/spmT_0024.nii")
    if verbose: mroi.plot_feature('activ')
    return mroi

def test_mroi3(verbose=0):
    nim =  nifti.NiftiImage("/tmp/blob.nii")
    header = nim.header
    mroi = MultipleROI(header=header)
    mroi.as_multiple_balls(np.array([[-10.,0.,10.]]),np.array([7.0]))
    mroi.from_labelled_image("/tmp/blob.nii",np.arange(1,20))
    mroi.from_labelled_image("/tmp/blob.nii",np.arange(31,50))
    mroi.make_image("/tmp/mroi.nii")
    mroi.set_roi_feature_from_image('activ',"/tmp/spmT_0024.nii")
    if verbose: mroi.plot_feature('activ')
    valid = np.random.randn(mroi.k)>0.1
    mroi.clean(valid)
    return mroi
