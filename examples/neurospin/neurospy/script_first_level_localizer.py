"""
Script updated by Bertrand.
"""
import numpy as np
import os
import ScriptBVFunc
import Contrast
from configobj import ConfigObj
from os.path import join
import glob

#import dataEngine
#data = dataEngine.DataEngine()

#########
# Paths #
#########
# "fiac0",
DBPath = "/data/thirion/localizer"
Subjects = ["s12069"]#""s12277", "s12069","s12300","s12401","s12431","s12508","s12532","s12635","s12636","s12826","s12898","s12913","s12919","s12920"]
Acquisitions = ["acquisition"]
Sessions = ["loc1"]
fmri = "fMRI/"
t1mri = "anatomy"
glmDir = "glm"
contrastDir = "Contrast"
minfDir = "/"#"Minf"

#######################
# General Information #
#######################

TR = 2.4
nbFrames = 128

Conditions = [ 'damier_H', 'damier_V', 'clicDaudio', 'clicGaudio', 
'clicDvideo', 'clicGvideo', 'calculaudio', 'calculvideo', 'phrasevideo', 
'phraseaudio' ]

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
infThreshold = 0.2
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
    SubjectPath = os.sep.join((DBPath, s))
    for a in Acquisitions:
        fmriFiles = {}
        for sess in Sessions:
            fmriPath = os.sep.join((SubjectPath, fmri, a, sess))
            fmriFiles[sess] = glob.glob(join(fmriPath,'S*.nii'))
            #fmriFile = data.findFiles(fmriPath, 'S*.nii', 1, 0)[0]
            #fmriFiles[sess] = unicode(fmriFile)
            
    ########################
    # First Level analysis #
    ########################
    for a in Acquisitions:
        #paradigmFile = data.findFiles(os.sep.join((SubjectPath, fmri, a, minfDir)), "paradigm.csv", 1, 0)[0]
        paradigmFile = join(os.sep.join((SubjectPath, fmri, a, minfDir)), "paradigm.csv")
        if not os.path.isfile(paradigmFile):
            raise ValueError,"paradigm file %s not found" %paradigmFile
        
        ## Creating and initialising Misc info file with the conditions
        ## for the design Matrix creation
        miscFile = os.sep.join((SubjectPath, fmri, a, minfDir, "misc_info.con"))
        misc=ConfigObj(miscFile)
        misc["sessions"] = Sessions
        misc["tasks"] = Conditions
        misc.write()

        for sess in Sessions:
            # Creating Design Matrix
            designPath = os.sep.join((SubjectPath, fmri, a, glmDir, sess))
            if not os.path.exists(designPath):
                os.makedirs(designPath)
            designFile = os.sep.join((designPath, "design_mat.csv"))
            ScriptBVFunc.DesignMatrix(nbFrames, paradigmFile, miscFile, TR, designFile, sess, hrfType, drift, drift_matrix, poly_order, cos_FreqCut, FIR_order, FIR_length)
        
        
        ## Computing Mask
        print "Computing Mask"
        maskFile = os.sep.join((SubjectPath, fmri, a, minfDir, "mask.img"))
        wc = join(fmriPath,'S*.nii') 
        ScriptBVFunc.ComputeMask(glob.glob(wc), maskFile, infThreshold, supThreshold)
        #ScriptBVFunc.ComputeMask(fmriFiles, maskFile, infThreshold, supThreshold)
        
        ## Creating Contrast File
        print "Creating Contrasts"
        contrast = Contrast.ContrastList(miscFile)
        d = contrast.dic
        ## Begin contrast declaration
        d["audio"] = d["clicDaudio"] + d["clicGaudio"] + d["calculaudio"] + d["phraseaudio"]
        d["video"] = d["clicDvideo"] + d["clicGvideo"] + d["calculvideo"] + d["phrasevideo"]
        d["left"] = d["clicGaudio"] + d["clicGvideo"]
        d["right"] = d["clicDaudio"] + d["clicDvideo"] 
        d["computation"] = d["calculaudio"] +d["calculvideo"]
        d["sentences"] = d["phraseaudio"] + d["phrasevideo"]
        d["H-V"] = d["damier_H"] - d["damier_V"]
        d["V-H"] =d["damier_V"] - d["damier_H"]
        d["left-right"] = d["left"] - d["right"]
        d["right-left"] = d["right"] - d["left"]
        d["audio-video"] = d["audio"] - d["video"]
        d["video-audio"] = d["video"] - d["audio"]
        d["computation-sentences"] = d["computation"] - d["sentences"]
        d["reading-visual"] = d["sentences"]*2 - d["damier_H"] - d["damier_V"]

        ## End contrast declaration
        contrastFile = os.sep.join((SubjectPath, fmri, a, minfDir, "contrast.con"))
        contrast.save_dic(contrastFile)
        ## Fitting glm
        glms = {}
        for sess in Sessions:
            print "Fitting GLM for session : %s" % sess
            glmPath = os.sep.join((SubjectPath, fmri, a, glmDir, sess))
            HDF5File = os.sep.join((glmPath, "vba.npz"))
            configFile = os.sep.join((glmPath, "vba_config.con"))
            designPath = os.sep.join((SubjectPath, fmri, a, glmDir, sess))
            designFile = os.sep.join((designPath, "design_mat.csv"))
            if os.path.exists(designFile):
                #fname = np.sort(glob.glob(wc))
                #fname = [f for f in fname]
                ScriptBVFunc.GLMFit(fmriFiles[sess], designFile, maskFile, HDF5File, configFile, fit_algo)
                glms[sess] = {}
                glms[sess]["HDF5FilePath"] = HDF5File
                glms[sess]["ConfigFilePath"] = configFile
        ## Compute Contrasts
        paths["Contrasts_path"] = os.sep.join((SubjectPath, fmri, a, glmDir, contrastDir))
        if not os.path.exists(paths["Contrasts_path"]):
            os.makedirs(paths["Contrasts_path"])
        print "Computing contrasts"
        ScriptBVFunc.ComputeContrasts(contrastFile, miscFile, glms, ScriptBVFunc.saveall, save_mode, paths = paths)
            
