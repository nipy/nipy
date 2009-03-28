import types, glob, exceptions
import fff2.glm as GLM
#
from datamind.core import DF
from datamind.core import url as URL
from datamind.io   import objIO

from soma import aims
from numpy import *
from neurospy.bvfunc import tio
import datamind.image.vol
import datamind.image.aimsutils

#
from datamind.stats.design import *

from configobj import ConfigObj

# ------------------------------------------------------------------------------
# Voxel Based Analysis
# ------------------------------------------------------------------------------
class CorticalGLM:
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
    N.max(N.abs(t-t2))
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
            #for k in kw.keys(): self.__dict__[k]=kw[k]
            if not hasattr(self, "mri_names"):
                self.mri_names = obj[:,0].tolist()
                obj = obj[:, 1:]
            if create_design_mat:
                self.yX=buildDesignMat(obj,X_formula,X_interaction)
                #y=DF([[v] for v in Y_url],colnames=["url"])
                #self.yX=y.concatenate(X,axis=1)
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
        Left_tex=self.getLeft_tex()
        Right_tex=self.getRight_tex()
        X=N.asarray(self.yX)
        if not hasattr(self,"method"):
            self.method="ols"
            self.model='spherical'
        self._left_model=GLM.glm()
        self._left_model.fit(Left_tex, X, model = self.model, method = self.method)
        self._right_model=GLM.glm()
        self._right_model.fit(Left_tex, X, model = self.model, method = self.method)
            # Save results on disk?
    # --------------------------------------------------------------------------
    # Contrasts & test
    def contrast(self,c,type='t'):
        """
        See fff.glm.glm.contrast
        """
        self.getModel()
        self._left_con = self._left_model.contrast(c,type=type)
        self._right_con = self._right_model.contrast(c,type=type)
    def test_left(self, zscore=False):
        """
        See fff.glm.contrast.test
        """
        return self._left_con.test(zscore=zscore)

    def test_right(self, zscore=False):
        """
        See fff.glm.contrast.test
        """
        return self._right_con.test(zscore=zscore)

    # --------------------------------------------------------------------------
    # accessors over structures & I/O
    # If a structure is absent try to load it, the same way it was saved
    def save(self,url=None):
        """
        Save the object, see datamind.io.objIO.writeObj()
        """
        if type(url) == dict:
            self._right_model.save(url["RightHDF5FilePath"])
            self._left_model.save(url["LeftHDF5FilePath"])
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
            #self.saveModel(url)
            url=URL.joinUrl(url,"model","h5")
            self.writeModel(url)
            objIO.writeObj(self,url)

    def load(self, dic):
        if type(dic) is dict:
            self.__dict__.update(ConfigObj(dic["ConfigFilePath"]).dict())
            self.yX = DF.read(self.DesignFilePath)
            self._left_model = GLM.load(dic["LeftHDF5FilePath"])
            self._right_model = GLM.load(dic["RightHDF5FilePath"])
        
    def getLeft_tex(self):
        """
        Return the Y array, read the volumes if required
        """
        if not hasattr(self,"_Left_tex"):
            if hasattr(self, "left_tex"):
                tex = aims.read(self.left_tex)
                temp = array(tex[0])
                n = range(1, tex.size())
                for t in n:
                    temp = vstack((temp, array(tex[t])))
                self._Left_tex = temp
            else:
                return None
        return self._Left_tex

    def getRight_tex(self):
        """
        Return the Y array, read the volumes if required
        """
        if not hasattr(self,"_Right_tex"):
            if hasattr(self, "right_tex"):
                tex = aims.read(self.right_tex)
                temp = array(tex[0])
                n = range(1, tex.size())
                for t in n:
                    temp = vstack((temp, array(tex[t])))
                self._Right_tex = temp
            else:
                return None
        return self._Right_tex

    def getModel(self):
        """
        Return beta, norm_var_beta, s2, a, dof
        See fff.glm.glm 
        """
        if hasattr(self,"_left_model"): return 0
        self.fit()
        
    def saveVol(self,obj,suffix=None,url=None):
        if url is None:
            url=URL.joinUrl(self.output_url,suffix,"tex")
        if size(obj.shape) == 1:
            T = aims.Texture_FLOAT(obj.astype('float32'))
        else:
            T = aims.TimeTexture_FLOAT()
            for i, n in enumerate(obj):
                T[i] = aims.Texture_FLOAT(n.astype('float32'))
        aims.write(T, url)
        
    def writeModel(self,file):
        """
        Deprecated Method. Use _left_model.save() or _right_model.save(). Now doing nothing at
        all.
        """
#         import tables
#         # Open a new empty HDF5 file
#         fileh = tables.openFile(file, mode = "w")
#         # Get the root group
#         lmodel = self._left_model
#         rmodel = self._right_model
#         root = fileh.root
#         lbeta = fileh.createArray (root, 'lbeta', lmodel.beta, "lBeta")
#         ls2   = fileh.createArray (root, 'ls2',   lmodel.s2, "ls2")
#         lnorm_var_beta= fileh.createArray (root,'lnorm_var_beta', lmodel.norm_var_beta,"lnorm_var_beta")

#         if lmodel.a != None:
#             la = fileh.createArray(root, 'la', lmodel.a, "la")
#         else:
#             root._v_attrs.la = lmodel.a
#         root._v_attrs.laxis=lmodel.axis
#         root._v_attrs.lnorm_var_beta_constant=lmodel.norm_var_beta_constant
#         root._v_attrs.ldof=lmodel.dof

#         rbeta = fileh.createArray (root, 'rbeta', rmodel.beta, "rBeta")
#         rs2   = fileh.createArray (root, 'rs2',   rmodel.s2, "rs2")
#         rnorm_var_beta= fileh.createArray (root,'rnorm_var_beta', rmodel.norm_var_beta,"rnorm_var_beta")

#         if rmodel.a != None:
#             ra = fileh.createArray(root, 'ra', rmodel.a, "ra")
#         else:
#             root._v_attrs.ra = rmodel.a
#         root._v_attrs.raxis = rmodel.axis
#         root._v_attrs.rnorm_var_beta_constant = rmodel.norm_var_beta_constant
#         root._v_attrs.rdof = rmodel.dof
#         fileh.close()

    def readModel(self,file):
        """
        Deprecated Method. Now doing nothing at all
        """
#         import tables
#         fileh = tables.openFile(file, mode = "r")
#         root = fileh.root
#         import fff.glm as GLM
#         lmodel=GLM.glm(None,None)
#         lmodel.beta                  =root.lbeta.read()
#         lmodel.s2                    =root.ls2.read()
#         lmodel.norm_var_beta         =root.lnorm_var_beta.read()
#         try:
#             lmodel.a                     =root._v_attrs.la
#         except:
#             lmodel.a                     =root.la.read()
#         lmodel.axis                  =root._v_attrs.laxis
#         lmodel.norm_var_beta_constant=root._v_attrs.lnorm_var_beta_constant
#         lmodel.dof                   =root._v_attrs.ldof

#         rmodel=GLM.glm(None,None)
#         rmodel.beta                  =root.rbeta.read()
#         rmodel.s2                    =root.rs2.read()
#         rmodel.norm_var_beta         =root.rnorm_var_beta.read()
#         try:
#             rmodel.a                     =root._v_attrs.ra
#         except:
#             rmodel.a                     =root.ra.read()
#         rmodel.axis                  =root._v_attrs.raxis
#         rmodel.norm_var_beta_constant=root._v_attrs.rnorm_var_beta_constant
#         rmodel.dof                   =root._v_attrs.rdof

#         fileh.close()
#         self._left_model = lmodel
#         self._right_model = rmodel

    # --------------------------------------------------------------------------
    # Utils urls
    def getFullYUrl(self,url):
        if hasattr(self,"Y_common_prefix_url"):
            return URL.joinUrl(self.Y_common_prefix_url,url)
        else:return url


