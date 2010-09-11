# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Script that perform the first-level analysis of a dataset of the FIAC
Last updated by B.Thirion

Author : Lise Favre, Bertrand Thirion, 2008-2009
"""
import os

from numpy import arange

from nipy.externals.configobj import ConfigObj
from nipy.neurospin import compute_mask_files
from nipy.neurospin.glm_files_layout import glm_tools, contrast_tools

# -----------------------------------------------------------
# --------- Set the paths -----------------------------------
#-----------------------------------------------------------

DBPath = "/volatile/thirion/db/word_reading"
Subjects = ["sbt0801651598-0002-00001-000160-01"]#, 
Acquisitions = ["default_acquisition"]
Sessions = ["session_01","session_02","session_03","session_04","session_05"]
model_id = "stimed"
fmri_wc = "asession_*_*.nii"

# ---------------------------------------------------------
# -------- General Information ----------------------------
# ---------------------------------------------------------

tr = 2.4
nb_frames = 154
frametimes = arange(nb_frames) * tr
Conditions = [ 'trial_%04d' %i for i in range(230) ]

# ---------------------------------------------------------
# ------ First level analysis parameters ---------------------
# ---------------------------------------------------------

#---------- Masking parameters 

infTh = 0.75
supTh = 0.95

#---------- Design Matrix

hrf_model = "Canonical"
drift_model = "Cosine"
hfcut = 128

#-------------- GLM options
# Possible choices : "Kalman_AR1", "Kalman", "Ordinary Least Squares"
fit_algo = "Ordinary Least Squares"#"Kalman_AR1"

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
    
    for a in Acquisitions:
        # step 1. set all the paths
        basePath = os.sep.join((DBPath, s, "fMRI", a))
        paths = glm_tools.generate_all_brainvisa_paths( basePath, Sessions, 
                                                        fmri_wc, model_id)  
          
        misc = ConfigObj(paths['misc'])
        misc["sessions"] = Sessions
        misc["tasks"] = Conditions
        misc["mask_url"] = paths['mask']
        misc.write()

        # step 2. Create one design matrix for each session
        design_matrices = {}
        for sess in Sessions:
            design_matrices[sess] = glm_tools.design_matrix(
                 paths['misc'], paths['dmtx'][sess], sess, paths['paradigm'],
                frametimes, hrf_model=hrf_model, drift_model=drift_model,
                hfcut=hfcut, model=model_id)
                        
        # step 3. Compute the Mask
        # fixme : it should be possible to provide a pre-computed mask
        print "Computing the Mask"
        mask_array = compute_mask_files( paths['fmri'].values()[0][0], 
                                         paths['mask'], True, infTh, supTh)
                                      
        # step 4. Creating functional contrasts
        print "Creating Contrasts"
        clist = contrast_tools.ContrastList(misc=ConfigObj(paths['misc']),
                                            model=model_id)
        contrast = clist.save_dic(paths['contrast_file'])
        CompletePaths = glm_tools.generate_brainvisa_ouput_paths( 
                        paths["contrasts"],  contrast)

        # step 5. Fit the  glm for each session
        glms = {}
        for sess in Sessions:
            print "Fitting GLM for session : %s" % sess
            glms[sess] = glm_tools.glm_fit(
                paths['fmri'][sess], design_matrices[sess],
                paths['glm_dump'][sess], paths['glm_config'][sess],
                fit_algo, paths['mask'])
            
        #step 6. Compute Contrasts
        print "Computing contrasts"
        # this one breaks the memory
        #contrast.pop('effect_of_interest')
        contrast["contrast"].remove('effect_of_interest')
        glm_tools.compute_contrasts(
            contrast, misc, CompletePaths, glms, model=model_id)


       
        
