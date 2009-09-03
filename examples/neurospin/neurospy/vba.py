import numpy as np
import types, glob, exceptions
from configobj import ConfigObj
import nifti
#
import nipy.neurospin.glm as GLM
#
from dataFrame import DF
import url as URL
import objIO

#from datamind.stats.design import *
#import soma.aims as aims

bnifti=True

# ------------------------------------------------------------------------------
# Voxel Based Analysis
# ------------------------------------------------------------------------------
class VBA:
    """
    Voxel Based Analysis, do the link between output of DB query results and FFF:
    - DB query results are tuples with url to images and regressor (Age,Sex, ...)
    - From those tuples do all the tedious job to call FFF routines fit/contrast/test
      - format the design matrix (code factor regressors)
      - manage mask
      - Save/reload beta, s2 and model parameters

    from datamind.core import DF
    from datamind.stats import VBA
    
    # Optional PATH to mask and common prefix path to images
    import os.path
    BASEPATH=os.path.join(os.environ['HOME'],"Data/07Deprim/")
    IMAGESPATH=BASEPATH+"IRM/Data/"
    MASKPATH=BASEPATH+"IRM/Mask/mask.img"
    
    tab=DF.read("tuples.csv")[:,['IRM_G_file', 'Response', 'Sex','Age']]
    print tab
    #IRM_G_file      Response        Sex     Age
    #smwc1S6182.img  N               2.0     45.0
    #smwc1S7302.img  N               2.0     44.0
    #....            .               ...     ....
    #smwc1S10140.img Y               1.0     38.0
    #smwc1S10291.img Y               2.0     49.5
    #smwc1E10590.img Y               2.0     39.5
    
    # If the table contain the full path, Y_common_prefix_url is optional
    m=VBA(tab,mask_url=MASKPATH,Y_common_prefix_url=IMAGESPATH)
    print m.yX
    #url             Response_N      Response_Y      Sex     Age
    #smwc1S6182.img  1.0             0.0             2.0     45.0
    #smwc1S7302.img  1.0             0.0             2.0     44.0
    #..............  ...             ...             ...     ....
    #smwc1S10291.img 0.0             1.0             2.0     49.5
    #smwc1E10590.img 0.0             1.0             2.0     39.5
    
    
    m.fit()
    m.contrast([1,-1,0,0])
    t, p=m.test()
    m.save("/tmp/test1")
    #ls /tmp/test1.*
    #/tmp/test1.minf  /tmp/test1.model.h5  /tmp/test1.yX.csv
    #
    # test1.minf     essentially contains urls (mask, output dir) 
    # test1.yX.csv   contains the design matrix & url to images
    # test1.model.h5 model parameters beta, s2 etc...
    
    m.saveVol(t,"t")
    
    m2=VBA("/tmp/test1")
    m2.contrast([1,-1,0,0])
    m2.save("/tmp/test2")
    t2, p2=m2.test()
    m2.saveVol(t2,"t")
    
    import numpy as N
    np.max(np.abs(t-t2))
    """
    # --------------------------------------------------------------------------
    # Constructor
    def __init__(self, obj=None, X_formula=None, X_interaction=None, create_design_mat = True, **kw):
        """
        if obj is a dataframe the first column is the url to the image and the
        others columns are the regressors
        if obj is a string it is a path to a file name "a .minf" file
    
        Optional args:
        X_formula: R like formula "sex+age+sex:age" (not implemented yet)
        X_interaction: True/False (not implemented yet)
        Y_common_prefix_url
        mask_url
        output_url
        """
        if isinstance(obj, DF):
            # Copy all optionnal argument in dict
            self.__dict__.update(kw)
            if not hasattr(self, "mri_names"):
                self.mri_names = obj[:,0].tolist()
                obj = obj[:, 1:]
            if create_design_mat:
                self.yX = buildDesignMat(obj,X_formula,X_interaction)
            else:
                self.yX=obj
            
        elif type(obj) is types.StringType:
            url=obj
            objIO.readObj(self,url)
        elif type(obj) is dict :
            self.load(obj)

    # --------------------------------------------------------------------------
    # Fit
    def fit(self):
        """
        See fff.glm.fit
        """
        ## Load Images => self.Y_arr & mask => self.mask_arr
        Y_arr=self.getY_arr()
        X=np.asarray(self.yX)
        if not hasattr(self,"method"):
            self.method="ols"
            self.model='spherical'
        self._model=GLM.glm()
        self._model.fit(Y_arr, X, method=self.method)
        del(self._Y_arr)
        del(Y_arr)

    # --------------------------------------------------------------------------
    # Contrasts & test
    def contrast(self,c,type='t'):
        """
        See fff.glm.glm.contrast
        """
        model=self.getModel()
        self._con = model.contrast(c,type=type)
    def test(self, zscore=False):
        """
        See fff.glm.contrast.test
        """
        return self._con.test(zscore=zscore)

    # --------------------------------------------------------------------------
    # accessors over structures & I/O
    # If a structure is absent try to load it, the same way it was saved
    def save(self,url=None):
        """
        Save the object, see datamind.io.objIO.writeObj()
        """
        if type(url) == dict:
            self._model.save(url["GlmDumpFile"])
            pythons = ConfigObj(url["ConfigFilePath"])
            for k in self.__dict__.keys():
                if k[0]=="_" or type(self.__dict__[k]) is types.InstanceType or isinstance(self.__dict__[k], DF):
                    continue
                else:
                    pythons[k]=self.__dict__[k]
            pythons["DesignFilePath"] = url["DesignFilePath"]
            pythons.write()
        else:
            self.output_url=url 
            url=URL.joinUrl(url,"model","h5")
            self._model.save(self._model,url)

    def load(self, dic):
        if type(dic) is dict:
            self.__dict__.update(ConfigObj(dic["ConfigFilePath"]).dict())
            self.yX = DF.read(self.DesignFilePath)
            self._model = GLM.load(dic["GlmDumpFile"])
        
    def getY_arr(self):
        """
        Return the Y array, read the volumes if required
        """
        if not hasattr(self,"_Y_arr"):
            mask_arr = self.getMask_arr()
            if type(self.mri_names) is list:
                Y_urls=[self.getFullYUrl(url) for url in self.mri_names]
            else:
                Y_urls=[self.getFullYUrl(self.mri_names)]

            temp = nifti.NiftiImage(str(Y_urls[0]))
            if mask_arr != None:
                vols = temp.asarray()[:, mask_arr]
            else:
                vols = temp.asarray()
            print "Read : %s" % Y_urls[0]
            self.data_size = vols.shape
            for url in Y_urls[1:]:
                temp = nifti.NiftiImage(str(url))
                if mask_arr != None:
                    vols = np.vstack(temp.asarray()[:, mask_arr])
                else:
                    vols = np.vstack(temp.asarray())

                print "Read : %s" % url
            self._Y_arr=vols
            del temp
            del vols
        return self._Y_arr

    def getMask_arr(self):
        """
        Return the mask array, read the volume if required
        If flat is True return the flattened mask
        """
        if not hasattr(self,"_mask_arr"):
            if not hasattr(self,"mask_url"): return None
            
            temp = nifti.NiftiImage(self.mask_url)
            self._mask_arr = temp.asarray()
            if np.size(self._mask_arr.shape) == 4:
                self._mask_arr=self._mask_arr[0]
            print "Read : %s" % self.mask_url
            self._mask_arr = self._mask_arr - self._mask_arr.min()
            self._mask_arr = self._mask_arr.astype('bool')
        return self._mask_arr

    def getModel(self):
        """
        Return beta, norm_var_beta, s2, a, dof
        See fff.glm.glm 
        """
        if hasattr(self,"_model"): return self._model
        try:
            url=URL.joinUrl(self.output_url,"model","h5")
            self._model=GLM.load(url)

        except exceptions.Exception, e:
            print "Model could not be loaded, call fit",e
            self.fit()
        return self._model
        
    def saveVol(self,obj,suffix=None,url=None):
        if url is None:
            url=URL.joinUrl(self.output_url,suffix,"img")
        if isinstance(obj, np.ndarray):
            obj=self.arr2vol(obj)
        nifti.NiftiImage(obj,url)
        
    def writeModel(self,model,file):
        """
        Deprecated Method. Use _model.save() instead
        """
        import tables
        # Open a new empty HDF5 file
        fileh = tables.openFile(file, mode = "w")
        # Get the root group
        root = fileh.root
        beta = fileh.createArray (root, 'beta', model.beta, "Beta")
        s2   = fileh.createArray (root, 's2',   model.s2, "s2")
        norm_var_beta= fileh.createArray (root,'norm_var_beta', model.norm_var_beta,"norm_var_beta")

        if model.a != None:
            a = fileh.createArray(root, 'a', model.a, "a")
        else:
            root._v_attrs.a=model.a
        root._v_attrs.axis=model.axis
        root._v_attrs.norm_var_beta_constant=model.norm_var_beta_constant
        root._v_attrs.dof=model.dof
        fileh.close()

    def readModel(self,file):
        """
        Deprecated Method.
        """
        import tables
        fileh = tables.openFile(file, mode = "r")
        root = fileh.root
        import fff.glm as GLM
        model=GLM.glm(None,None)
        model.beta                  =root.beta.read()
        model.s2                    =root.s2.read()
        model.norm_var_beta         =root.norm_var_beta.read()
        try:
            model.a                     =root._v_attrs.a
        except:
            model.a                     =root.a.read()
        model.axis                  =root._v_attrs.axis
        model.norm_var_beta_constant=root._v_attrs.norm_var_beta_constant
        model.dof                   =root._v_attrs.dof
        fileh.close()
        return model

    # --------------------------------------------------------------------------
    # Utils volumes <=> arrays
    def getRefBlankVol(self,voltype="DOUBLE"):
        """
        Return a blank (float) of os the same dimension than the input volumes.
        Use it to store t-map etc...
        """
        if not hasattr(self,"_refBlankVol"):
            print type(self.mri_names) == str
            print self.mri_names
            if type(self.mri_names) == str or type(self.mri_names) == unicode:
                url=[self.getFullYUrl(self.mri_names)]
                print url
                print type(url)
            else:
                url=[self.getFullYUrl(self.mri_names[0])]
            vol=datamind.image.vol.readVolumes(url)[0]
            self._refBlankVol=datamind.image.aimsutils.vol_convert(vol,desttype=voltype)
        return(datamind.image.aimsutils.vol_clone_init(vol=self._refBlankVol,val=0))


    def arr2vol(self,arr):
        """
        array to volume convertion, taking in account a mask if exists
        """
        vol=self.getRefBlankVol()
        vol_arr=vol.asarray()
        mask_arr=self.getMask_arr()
        if not mask_arr is None:
            for volume in vol_arr:
                # Iterate over the first dimension
                volume[mask_arr]=arr
        else:
            vol_arr=arr
        return vol
    def vol2arr(self,vol):
        """
        volume to array convertion, taking in account a mask if exists
        """
        vol_arr = vol.asarray()
        mask_arr=self.getMask_arr()
        if not mask_arr is None:
            arr=vol_arr[mask_arr]
        else:
            arr=vol_arr
        return arr
    # --------------------------------------------------------------------------
    # Utils urls
    def getFullYUrl(self,url):
        if hasattr(self,"Y_common_prefix_url"):
            return URL.joinUrl(self.Y_common_prefix_url,url)
        else:return url


