"""
Script that perform the first-level analysis of a dataset of the FIAC
Last updated by B.Thirion

Author : Lise Favre, Bertrand Thirion, 2008-2009
"""

import os
from configobj import ConfigObj
from nipy.neurospin.utils.mask import compute_mask_files
import glm_tools, contrast_tools


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
modelDir = "default"
fmri_wc = "swra*.nii.gz"

# ---------------------------------------------------------
# -------- General Information ----------------------------
# ---------------------------------------------------------

TR = 2.5
nbFrames = 191
# NB : it is the same for all sessions, acqusitions, subjects

Conditions = ["SSt-SSp", "SSt-DSp", "DSt-SSp", "DSt-DSp", "FirstSt"]

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
cos_FreqCut = 128

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

    for a in Acquisitions:
        # step 1. set all the paths
        basePath = os.sep.join((DBPath, s, "fMRI", a))
        paths = glm_tools. generate_all_brainvisa_paths( basePath, Sessions, 
                                                        fmri_wc, modelDir) 
    
        misc = ConfigObj(paths['misc'])
        misc["sessions"] = Sessions
        misc["tasks"] = Conditions
        misc["mask_url"] = paths["mask"]
        misc.write()

        # step 2. Create one design matrix for each session
        design_matrices={}
        for sess in Sessions:
            design_matrices[sess] =\
               glm_tools.DesignMatrix( nbFrames, paths['paradigm'],
                                       paths['misc'], TR, paths['dmtx'][sess],
                                       sess, hrfType=hrfType, drift=drift,  
                                       cos_FreqCut=cos_FreqCut, model=modelDir)

        # step 3. Compute the Mask
        # fixme : it should be possible to provide a pre-computed mask
        print "Computing the Mask"
        mask_array = compute_mask_files(paths['fmri'].values()[0][0],
                                        paths['mask'], True, infTh, supTh)
        
        # step 4. Create Contrast Files
        print "Creating Contrasts"
        clist = contrast_tools.ContrastList(misc=misc)
        d = clist.dic
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
        contrast = clist.save_dic(paths['contrast_file'])
        CompletePaths = glm_tools.generate_brainvisa_ouput_paths( 
            paths["contrasts"],  contrast)
        
        # step 5. Fit the  glm for each session 
        glms = {}
        for sess in Sessions:
            print "Fitting GLM for session : %s" % sess
            glms[sess] = glm_tools.GLMFit(
                paths['fmri'][sess], design_matrices[sess] ,
                paths['glm_dump'][sess], paths['glm_config'][sess],
                fit_algo, paths['mask'])
         
        #6. Compute the Contrasts
        print "Computing contrasts"
        glm_tools.ComputeContrasts(contrast, misc, CompletePaths,
                                   glms, save_mode,
                                  threshold=3.0, cluster=10, method='None')  
        
            
