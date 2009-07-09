"""
Script that perform the first-level analysis of a dataset of the FIAC
Last updated by B.Thirion

Author : Lise Favre, Bertrand Thirion, 2008-2009
"""
import os
from configobj import ConfigObj
from os.path import join
import glob

import GLMTools
import Contrast

# -----------------------------------------------------------
# --------- Set the paths -----------------------------------
#-----------------------------------------------------------

DBPath = "/volatile/thirion/Localizer"
Subjects = ["s12069"]#""s12277", "s12069","s12300","s12401","s12431","s12508","s12532","s12635","s12636","s12826","s12898","s12913","s12919","s12920"]
Acquisitions = ["acquisition"]
Sessions = ["loc1"]
fmri = "fMRI/"
t1mri = "anatomy"
glmDir = "glm"
contrastDir = "Contrast"
minfDir = "Minf"

# ---------------------------------------------------------
# -------- General Information ----------------------------
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# ------ First level analysis parameters ---------------------
# ---------------------------------------------------------

#---------- Masking parameters 
infTh = 0.2
supTh = 0.9

#---------- Design Matrix

# Possible choices for hrfType : "Canonical", \
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

#-------------- GLM options
# Possible choices : "Kalman_AR1", "Kalman", "Ordinary Least Squares"
fit_algo = "Kalman_AR1"

#-------------- Contrast Options
# Possible choices : "Contrast Name" or "Contrast Number"
save_mode = "Contrast Name"

# ------------------------------------------------------------------
# Launching Pipeline on all subjects, all acquisitions, all sessions 
# -------------------------------------------------------------------

# fixme : normally all that part should be data-independent,
# i.e. a standard user should not have to look at it
# which means that the paths are completely set at that point

# Treat sequentially all subjects & acquisitions
for s in Subjects:
    print "Subject : %s" % s
    SubjectPath = os.sep.join((DBPath, s))
    
    for a in Acquisitions:

        #step 0. Get the fMRI data
        fmriFiles = {}
        for sess in Sessions:
            fmriPath = os.sep.join((SubjectPath, fmri, a, sess))
            fmriFiles[sess] = glob.glob(join(fmriPath,'S*.nii'))
  
        # step 1. get the paradigm definition and create misc info file
        miscPath = os.sep.join((SubjectPath, fmri, a, minfDir))
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
            designPath = os.sep.join((SubjectPath, fmri, a, glmDir, sess))
            if not os.path.exists(designPath):
                os.makedirs(designPath)
            designFile = os.sep.join((designPath, "design_mat.csv"))
            GLMTools.DesignMatrix(nbFrames, paradigmFile, miscFile,
                                      TR, designFile, sess, DmtxParam)
        
        # step3. Compute the Mask
        # fixme : it should be possible to provide a pre-computed mask
        print "Computing the Mask"
        maskFile = os.sep.join((SubjectPath, fmri, a, minfDir, "mask.img"))
        #wc = join(fmriPath,'S*.nii') 
        #GLMTools.ComputeMask(glob.glob(wc), maskFile, infTh, supTh)
        GLMTools.ComputeMask(fmriFiles.values()[0][0], maskFile, infTh, supTh)
        
        # step 4. Creating Contrast File
        print "Creating Contrasts"
        contrast = Contrast.ContrastList(miscFile)
        d = contrast.dic
        d["audio"] = d["clicDaudio"] + d["clicGaudio"] +\
                     d["calculaudio"] + d["phraseaudio"]
        d["video"] = d["clicDvideo"] + d["clicGvideo"] + \
                     d["calculvideo"] + d["phrasevideo"]
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
        contrastFile = os.sep.join((SubjectPath, fmri, a, minfDir,
                                    "contrast.con"))
        contrast.save_dic(contrastFile)

        # step 5. Fit the  glm for each session
        glms = {}
        for sess in Sessions:
            print "Fitting GLM for session : %s" % sess
            glmPath = os.sep.join((SubjectPath, fmri, a, glmDir, sess))
            GlmDumpFile = os.sep.join((glmPath, "vba.npz"))
            configFile = os.sep.join((glmPath, "vba_config.con"))
            designPath = os.sep.join((SubjectPath, fmri, a, glmDir, sess))
            designFile = os.sep.join((designPath, "design_mat.csv"))
            if os.path.exists(designFile):
                GLMTools.GLMFit(fmriFiles[sess], designFile, maskFile,
                                GlmDumpFile, configFile, fit_algo)
                glms[sess] = {}
                glms[sess]["GlmDumpFile"] = GlmDumpFile
                glms[sess]["ConfigFilePath"] = configFile
        #6. Compute Contrasts
        print "Computing contrasts"
        paths["Contrasts_path"] = os.sep.join((SubjectPath, fmri, a,
                                               glmDir, contrastDir))
        if not os.path.exists(paths["Contrasts_path"]):
            os.makedirs(paths["Contrasts_path"])
    
        GLMTools.ComputeContrasts(contrastFile, miscFile, glms,\
                                  save_mode, paths = paths)
            
