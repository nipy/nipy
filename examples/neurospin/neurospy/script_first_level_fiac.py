"""
Script updated by Bertrand.
"""

import os
import ScriptBVFunc
import Contrast
from configobj import ConfigObj

import dataEngine
data = dataEngine.DataEngine()

#########
# Paths #
#########
Sessions = ["loc1"]
fmri = "fMRI/"
t1mri = "anatomy"
glmDir = "glm"
contrastDir = "Contrast"
minfDir = "/"#"Minf"


DBPath = "/data/thirion/"
#DBPath ="/neurospin/lnao/Panabase/data_fiac/fiac_fsl/"
Subjects = [ "fiac1"]#, "fiac2", "fiac3", "fiac4", "fiac6", "fiac8", "fiac9", "fiac10", "fiac11", "fiac12", "fiac13", "fiac14", "fiac15"]
Acquisitions = ["acquisition"]
Sessions = ["fonc1", "fonc2", "fonc3", "fonc4"]
fmri = "fMRI"
t1mri = "t1mri/default_acquisition"
glmDir = "glm"
contrastDir = "Contrast"
minfDir = "Minf"

#######################
# General Information #
#######################

TR = 2.5
nbFrames = 191

Conditions = ["SSt-SSp", "SSt-DSp", "DSt-SSp", "DSt-DSp", "FirstSt"]

paths = {}
paths["Z map"] = "z_map"
paths["Student-t tests"] = "T_map"
paths["Fisher tests"] = "F_map"
paths["Residual variance"] = "ResMS"
paths["contrast definition"] = "con"

################################
# First level analysis options #
################################

## Compute Mask
infThreshold = 0.4
supThreshold = 0.9

## Design Matrix

# Possible choices for hrfType : "Canonical", "Canonical With Derivative" or "FIR Model"
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

## GLM options
# Possible choices : "Kalman_AR1", "Kalman", "Ordinary Least Squares"
fit_algo = "Kalman_AR1"

## Contrast Options
# Possible choices : "Contrast Name" or "Contrast Number"
save_mode = "Contrast Name"

#####################################################################
# Launching Pipeline on all subjects, all acquisitions, all session #
#####################################################################

for s in Subjects:
    print "Subject : %s" % s
    sPath = os.sep.join((DBPath, s))
    for a in Acquisitions:
        fmriFiles = {}
        for sess in Sessions:
            fmriPath = os.sep.join((sPath, fmri, a, sess))
            print fmriPath
            fmriFile = data.findFiles(fmriPath, 's*%s*.nii.gz' % sess, 1, 0)[0]
            # Adding fmri to list for masking
            fmriFiles[sess] = unicode(fmriFile)
    ########################
    # First Level analysis #
    ########################
    for a in Acquisitions:
        paradigmFile = data.findFiles(os.sep.join((sPath, fmri, a, minfDir)), "paradigm.csv", 1, 0)[0]
        miscFile = os.sep.join((sPath, fmri, a, minfDir, "misc_info.con"))
        ## Creating and initialising Misc info file with the conditions
        ## for the design Matrix creation
        misc=ConfigObj(miscFile)
        misc["sessions"] = Sessions
        misc["tasks"] = Conditions
        misc.write()
        
        for sess in Sessions:
            # Creating Design Matrix
            designPath = os.sep.join((sPath, fmri, a, glmDir, sess))
            if not os.path.exists(designPath):
                os.makedirs(designPath)
            designFile = os.sep.join((designPath, "design_mat.csv"))
            ScriptBVFunc.DesignMatrix(nbFrames, paradigmFile, miscFile, TR, designFile, sess, hrfType, drift, drift_matrix, poly_order, cos_FreqCut, FIR_order, FIR_length)
        
        ## Computing Mask
        print "Computing Mask"
        maskFile = os.sep.join((sPath, fmri, a, minfDir, "mask.img"))
        ScriptBVFunc.ComputeMask([str(fmriFiles.values()[0])], maskFile, infThreshold, supThreshold)
        ## Creating Contrast File
        print "Creating Contrasts"
        contrast = Contrast.ContrastList(miscFile)
        d = contrast.dic
        ## To add a contrast :
        # d["new_contrast"] = d["contrast1"] + d["contrast2"]
        # d["new_contrast"] = d["contrast1"] - d["contrast2"]
        # d["new_contrast"] = d["contrast1"] * d["contrast2"]
        # d["new_contrast"] = d["contrast1"] * 2
        ## Be careful not to do :
        # d["new_contrast"] = 3 * d["contrast1"]
        # This won't work
        ## Begin contrast declaration
        d["SStSSp_minus_DStDSp"] = d["SSt-SSp"] - d["DSt-DSp"]
        d["DStDSp_minus_SStSSp"] = d["DSt-DSp"] - d["SSt-SSp"]
        d["DSt_minus_SSt"] = d["DSt-SSp"] + d["DSt-DSp"] - d["SSt-SSp"] - d["SSt-DSp"]
        d["DSp_minus_SSp"] = d["DSt-DSp"] - d["DSt-SSp"] - d["SSt-SSp"] + d["SSt-DSp"]
        d["DSt_minus_SSt_for_DSp"] = d["DSt-DSp"] - d["SSt-DSp"]
        d["DSp_minus_SSp_for_DSt"] = d["DSt-DSp"] - d["DSt-SSp"]
        if d.has_key("FirstSt"):
            d["Deactivation"] = d["FirstSt"] - d["SSt-SSp"] - d["DSt-DSp"] - d["DSt-DSp"] - d["SSt-SSp"]
        else:
            d["Deactivation"] = (d["SSt-SSp"] * -1) - d["DSt-DSp"] - d["DSt-DSp"] - d["SSt-SSp"]
        ## End contrast declaration
        contrastFile = os.sep.join((sPath, fmri, a, minfDir, "contrast.con"))
        contrast.save_dic(contrastFile)
        ## Fitting glm
        glms = {}
        for sess in Sessions:
            print "Fitting GLM for session : %s" % sess
            glmPath = os.sep.join((sPath, fmri, a, glmDir, sess))
            HDF5File = os.sep.join((glmPath, "vba.npz"))
            configFile = os.sep.join((glmPath, "vba_config.con"))
            designPath = os.sep.join((sPath, fmri, a, glmDir, sess))
            designFile = os.sep.join((designPath, "design_mat.csv"))
            if os.path.exists(designFile):
                ScriptBVFunc.GLMFit(fmriFiles[sess], designFile, maskFile, HDF5File, configFile, fit_algo)
                glms[sess] = {}
                glms[sess]["HDF5FilePath"] = HDF5File
                glms[sess]["ConfigFilePath"] = configFile
        ## Compute Contrasts
        paths["Contrasts_path"] = os.sep.join((sPath, fmri, a, glmDir, contrastDir))
        if not os.path.exists(paths["Contrasts_path"]):
            os.makedirs(paths["Contrasts_path"])
        print "Computing contrasts"
        ScriptBVFunc.ComputeContrasts(contrastFile, miscFile, glms, ScriptBVFunc.saveall, save_mode, paths = paths)
            
