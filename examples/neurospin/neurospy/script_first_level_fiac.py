"""
Script that perform the first-level analysis of a dataset of the FIAC
Last updated by B.Thirion

Author : Lise Favre, Bertrand Thirion, 2008-2009
"""

import os
from configobj import ConfigObj
import glob
from os.path import join

import GLMTools
import Contrast

# -----------------------------------------------------------
# --------- Set the paths -----------------------------------
#-----------------------------------------------------------

DBPath ="/neurospin/lnao/Panabase/data_fiac/fiac_fsl/"
Subjects = [ "fiac1"]
#Subjects = ["fiac2", "fiac3", "fiac4", "fiac6", "fiac8",\
#            "fiac9", "fiac10", "fiac11", "fiac12", "fiac13",\
#            "fiac14", "fiac15"]

Acquisitions = ["acquisition"]
Sessions = ["fonc1", "fonc2", "fonc3", "fonc4"]
fmri = "fMRI"
t1mri = "t1mri/default_acquisition"
glmDir = "glm"
contrastDir = "Contrast"
minfDir = "Minf"

# ---------------------------------------------------------
# -------- General Information ----------------------------
# ---------------------------------------------------------

TR = 2.5
nbFrames = 191
# NB : it is the same for all sessions, acqusitions, subjects

Conditions = ["SSt-SSp", "SSt-DSp", "DSt-SSp", "DSt-DSp", "FirstSt"]

paths = {}
paths["Z map"] = "z_map"
paths["Student-t tests"] = "T_map"
paths["Fisher tests"] = "F_map"
paths["Residual variance"] = "ResMS"
paths["contrast definition"] = "con"

# ---------------------------------------------------------
# ------ First level analysis parameters ---------------------
# ---------------------------------------------------------

#---------- Masking parameters 
infTh = 0.4
supTh = 0.9

#---------- Design Matrix

# Possible choices for hrfType : "Canonical",
# "Canonical With Derivative" or "FIR Model"
hrfType = "Canonical"

# Possible choices for drift : "Blank", "Cosine", "Polynomial"
drift = "Cosine"

# If drift is "Polynomial"
poly_order = 2

# If drift is "Cosine"
cos_FreqCut = 128

# If hrfType is "FIR Model"
FIR_order = 1
FIR_length = 1

# If the following in not none it will be considered to be the drift
drift_matrix = None

DmtxParam= {}
DmtxParam["hrfType"] = hrfType
DmtxParam["drift"] = drift
DmtxParam["poly_order"] = poly_order
DmtxParam["cos_FreqCut"] = cos_FreqCut
DmtxParam["FIR_order"] = FIR_order
DmtxParam["FIR_length"] = FIR_length
DmtxParam["drift_matrix"] = drift_matrix

#--------- GLM options
# Possible choices : "Kalman_AR1", "Kalman", "Ordinary Least Squares"
fit_algo = "Kalman_AR1"

#--------- Contrast Options
# Possible choices : "Contrast Name" or "Contrast Number"
save_mode = "Contrast Name"

# ------------------------------------------------------------------
# Launching Pipeline on all subjects, all acquisitions, all sessions 
# -------------------------------------------------------------------

# main loop on subjects
for s in Subjects:
    print "Subject : %s" % s
    sPath = os.sep.join((DBPath, s))

    for a in Acquisitions:
        
        #step 0. Get the fMRI data
        fmriFiles = {}
        for sess in Sessions:
            fmriPath = os.sep.join((sPath, fmri, a, sess))
            fmriFiles[sess] = glob.glob(join(fmriPath,'swra*%s*.nii.gz' %sess))
             
        # step 1. get the paradigm definition and create misc info file
        miscPath = os.sep.join((sPath, fmri, a, minfDir))
        paradigmFile = os.sep.join((miscPath, "paradigm.csv"))
        if not os.path.isfile(paradigmFile):
            raise ValueError,"paradigm file %s not found" %paradigmFile

        miscFile = os.sep.join((miscPath, "misc_info.con"))
        misc = ConfigObj(miscFile)
        misc["sessions"] = Sessions
        misc["tasks"] = Conditions
        misc.write()

        
        # step 2. Create one design matrix for each session
        for sess in Sessions:
            # Creating Design Matrix
            designPath = os.sep.join((sPath, fmri, a, glmDir, sess))
            if not os.path.exists(designPath):
                os.makedirs(designPath)
            designFile = os.sep.join((designPath, "design_mat.csv"))
            GLMTools.DesignMatrix(nbFrames, paradigmFile, miscFile,
                                  TR, designFile, sess, DmtxParam)
        """ 
        # step 3. Compute the mask
        print "Computing the mask"
        maskFile = os.sep.join((sPath, fmri, a, minfDir, "mask.img"))
        GLMTools.ComputeMask(fmriFiles.values()[0][0], maskFile,
                             infTh, supTh)
        """
        
        # step 4. Create Contrast Files
        print "Creating Contrasts"
        contrast = Contrast.ContrastList(miscFile)
        d = contrast.dic
        d["SStSSp_minus_DStDSp"] = d["SSt-SSp"] - d["DSt-DSp"]
        d["DStDSp_minus_SStSSp"] = d["DSt-DSp"] - d["SSt-SSp"]
        d["DSt_minus_SSt"] = d["DSt-SSp"] + d["DSt-DSp"]\
                             - d["SSt-SSp"] - d["SSt-DSp"]
        d["DSp_minus_SSp"] = d["DSt-DSp"] - d["DSt-SSp"]\
                             - d["SSt-SSp"] + d["SSt-DSp"]
        d["DSt_minus_SSt_for_DSp"] = d["DSt-DSp"] - d["SSt-DSp"]
        d["DSp_minus_SSp_for_DSt"] = d["DSt-DSp"] - d["DSt-SSp"]
        if d.has_key("FirstSt"):
            d["Deactivation"] = d["FirstSt"] - d["SSt-SSp"]\
                                - d["DSt-DSp"] - d["DSt-DSp"] - d["SSt-SSp"]
        else:
            d["Deactivation"] = (d["SSt-SSp"] * -1)\
                                - d["DSt-DSp"] - d["DSt-DSp"] - d["SSt-SSp"]
        contrastFile = os.sep.join((sPath, fmri, a, minfDir, "contrast.con"))
        contrast.save_dic(contrastFile)

        
        # step 5. Fit the  glm for each session 
        glms = {}
        for sess in Sessions:
            print "Fitting GLM for session : %s" % sess
            glmPath = os.sep.join((sPath, fmri, a, glmDir, sess))
            GlmDumpFile = os.sep.join((glmPath, "vba.npz"))
            configFile = os.sep.join((glmPath, "vba_config.con"))
            designPath = os.sep.join((sPath, fmri, a, glmDir, sess))
            designFile = os.sep.join((designPath, "design_mat.csv"))
            if os.path.exists(designFile):
                #GLMTools.GLMFit(fmriFiles[sess], designFile, maskFile,
                #                    GlmDumpFile, configFile, fit_algo)
                glms[sess] = {}
                glms[sess]["GlmDumpFile"] = GlmDumpFile
                glms[sess]["ConfigFilePath"] = configFile

        
        #6. Compute the Contrasts
        print "Computing contrasts"
        paths["Contrasts_path"] = os.sep.join((sPath, fmri, a,
                                               glmDir, contrastDir))
        if not os.path.exists(paths["Contrasts_path"]):
            os.makedirs(paths["Contrasts_path"])
        
        GLMTools.ComputeContrasts(contrastFile, miscFile, glms,\
                                      save_mode, paths = paths)
            
