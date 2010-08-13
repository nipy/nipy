# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Script that perform the first-level analysis of a dataset of the FIAC
Last updated by B.Thirion

Author : Lise Favre, Bertrand Thirion, 2008-2010
"""

import os

from numpy import arange

from nipy.externals.configobj import ConfigObj
from nipy.neurospin import compute_mask_files
from nipy.neurospin.glm_files_layout import glm_tools, contrast_tools


# -----------------------------------------------------------
# --------- Set the paths -----------------------------------
#-----------------------------------------------------------

DBPath ="/neurospin/lnao/Panabase/data_fiac/fiac_fsl/"
Subjects = [ "fiac1"]
#Subjects = ["fiac2", "fiac3", "fiac4", "fiac6", "fiac8",\
#            "fiac9", "fiac10", "fiac11", "fiac12", "fiac13",\
#            "fiac14", "fiac15"]
Acquisitions = ["acquisition"]
Sessions = ["fonc1", "fonc2"]#["fonc3", "fonc4"]
model_id = "default"
fmri_wc = "swra*.nii.gz"

# ---------------------------------------------------------
# -------- General Information ----------------------------
# ---------------------------------------------------------

tr = 2.5
nb_frames = 191
frametimes = tr * arange(nb_frames)

Conditions = ["SSt-SSp", "SSt-DSp", "DSt-SSp", "DSt-DSp", "FirstSt"]

# ---------------------------------------------------------
# ------ First level analysis parameters ---------------------
# ---------------------------------------------------------

#---------- Masking parameters 
infTh = 0.4
supTh = 0.9

#---------- Design Matrix

# hrf model, to be chosen among "Canonical",
# "Canonical With Derivative" or "FIR Model"
hrf_model = "Canonical With Derivative"

# Possible choices for drift : "Blank", "Cosine", "Polynomial"
drift_model = "Cosine"
hfcut = 128

#--------- GLM options
# Possible choices : "Kalman_AR1", "Kalman", "Ordinary Least Squares"
fit_algo = "Kalman_AR1"


#####################################################################
# Launching Pipeline on all subjects, all acquisitions, all sessions 
#####################################################################

# main loop on subjects
for s in Subjects:
    print "Subject : %s" % s

    for a in Acquisitions:
        # step 1. set all the paths
        basePath = os.sep.join((DBPath, s, "fMRI", a))
        paths = glm_tools. generate_all_brainvisa_paths( basePath, Sessions, 
                                                        fmri_wc, model_id) 
    
        misc = ConfigObj(paths['misc'])
        misc["sessions"] = Sessions
        misc["tasks"] = Conditions
        misc["mask_url"] = paths["mask"]
        misc.write()

        # step 2. Create one design matrix for each session
        design_matrices={}
        for sess in Sessions:
            design_matrices[sess] = glm_tools.design_matrix(
                paths['misc'], paths['dmtx'][sess], sess, paths['paradigm'],
                frametimes, hrf_model=hrf_model, drift_model=drift_model,
                hfcut=hfcut, model=model_id)

        # step 3. Compute the Mask
        # fixme : it should be possible to provide a pre-computed mask
        print "Computing the Mask"
        mask_array = compute_mask_files(paths['fmri'].values()[0][0],
                                        paths['mask'], False, infTh, supTh)
        
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
            glms[sess] = glm_tools.glm_fit(
                paths['fmri'][sess], design_matrices[sess] ,
                paths['glm_dump'][sess], paths['glm_config'][sess],
                fit_algo, paths['mask'])
         
        #6. Compute the Contrasts
        print "Computing contrasts"
        glm_tools.compute_contrasts(contrast, misc, CompletePaths,
                                   glms, model=model_id)  
        
            
