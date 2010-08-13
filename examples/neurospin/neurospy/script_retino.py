# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Script that performs the GLM analysis on the cortical surface
In order to obtain retinotopic maps

Author : Bertrand Thirion, 2010
"""

import os

import numpy as np

from nipy.externals.configobj import ConfigObj

from nipy.neurospin import compute_mask_files
from nipy.neurospin.glm_files_layout import glm_tools, contrast_tools, \
     cortical_glm


# -----------------------------------------------------------
# --------- Set the paths -----------------------------------
#-----------------------------------------------------------

DBPath = "/volatile/thirion/fs_db"
Subjects = ["bru3072"]
Acquisitions = [""]
Sessions = ["ima1","ima2","ima3","ima4"]
model_id = "default"

# choose volume-based or surface-based analysis
side = 'left'#'right'#'False'#
fmri_wc = "rima*.img"
if side=='left':
    fmri_wc = "left_tex*.tex"
elif side=='right':
    fmri_wc = "right_tex*.tex"
            
# ---------------------------------------------------------
# -------- General Information ----------------------------
# ---------------------------------------------------------

tr = 2.0
nb_frames = 130
frametimes = np.arange(nb_frames) * tr

period = 8.
r1 = np.sin( 2*np.pi*period/128 * np.arange(130))
r2 = np.cos( 2*np.pi*period/128 * np.arange(130))
reg_matrix = np.vstack((r1, r2)).T
Reg = {'ima1':['sin_wedge_pos','cos_wedge_pos'],
     'ima2':['sin_wedge_neg','cos_wedge_neg'],
     'ima3':['sin_ring_pos','cos_ring_pos'],
     'ima4':['sin_ring_neg','cos_ring_neg']}

AllReg = ['sin_wedge_neg','cos_wedge_neg','sin_wedge_pos','cos_wedge_pos',
          'sin_ring_pos','cos_ring_pos','sin_ring_neg','cos_ring_neg']

# ---------------------------------------------------------
# ------ First level analysis parameters ---------------------
# ---------------------------------------------------------

#---------- Masking parameters 
infTh = 0.7
supTh = 0.9

#---------- Design Matrix

# Possible choices for hrfType : "Canonical", \
# "Canonical With Derivative" or "FIR"
hrfType = "Canonical"

# Possible choices for drift : "Blank", "Cosine", "Polynomial"
drift_model = "Cosine"
hfcut = 80

#-------------- GLM options
# Possible choices : "Kalman_AR1", "Kalman", "Ordinary Least Squares"
fit_algo = "Kalman_AR1"

# ------------------------------------------------------------------
# Launching Pipeline on all subjects, all acquisitions, all sessions 
# -------------------------------------------------------------------

# Treat sequentially all subjects & acquisitions
for s in Subjects:
    print "Subject : %s" % s
    SubjectPath = os.sep.join((DBPath, s))
    
    for a in Acquisitions:

        # step 1. set all the paths
        basePath = os.sep.join((DBPath, s, "fmri", a))
        paths = glm_tools.generate_all_brainvisa_paths(
            basePath, Sessions, fmri_wc, model_id, paradigm_id=None)

        for sess in Sessions:
            paths['fmri'][sess] = paths['fmri'][sess][-nb_frames:]

        misc = ConfigObj(paths['misc'])
        misc["sessions"] = Sessions
        misc["tasks"] = AllReg
        misc['mask_url'] = paths['mask']
        misc[model_id]={}
        misc.write()
        
        # step 2. Create one design matrix for each session
        design_matrices = {}
        for sess in Sessions:
            design_matrices[sess] = glm_tools.design_matrix(
                paths['misc'], paths['dmtx'][sess], sess, None,
                frametimes, drift_model=drift_model, hfcut=hfcut,
                model=model_id, add_regs=reg_matrix, add_reg_names=Reg[sess] )


        # step 3. Compute the Mask
        # fixme : it should be possible to provide a pre-computed mask
        if side=='False':
            print "Computing the Mask"
            mask_array = compute_mask_files( paths['fmri'].values()[0][0], 
                                         paths['mask'], True, infTh, supTh)
                    
        # step 4. Creating Contrast File
        print "Creating Contrasts"
        clist = contrast_tools.ContrastList(
            misc=ConfigObj(paths['misc']), model=model_id) 
        d = clist.dic
        d["ring"] = d['effect_of_interest'].copy()
        d['ring']['ima1'] = np.zeros((2,6))
        d['ring']['ima2'] = np.zeros((2,6))
        d["wedge"] = d['effect_of_interest'].copy()
        d['wedge']['ima3'] = np.zeros((2,6))
        d['wedge']['ima4'] = np.zeros((2,6))
        contrast = clist.save_dic(paths['contrast_file'])
        if side=='False':
            CompletePaths = glm_tools.generate_brainvisa_ouput_paths( 
                paths["contrasts"],  contrast)
        else:
            CompletePaths = cortical_glm.generate_brainvisa_ouput_paths( 
                        paths["contrasts"],  contrast, side)
        
        # step 5. Fit the  glm for each session
        glms = {}
        for sess in Sessions:
            print "Fitting GLM for session : %s" % sess
            if side=='False':
                glms[sess] = glm_tools.glm_fit(
                    paths['fmri'][sess], design_matrices[sess],
                    paths['glm_dump'][sess], paths['glm_config'][sess],
                    fit_algo, paths['mask'])
            else:
                 glms[sess] = cortical_glm.glm_fit(
                    paths['fmri'][sess], design_matrices[sess],
                    paths['glm_dump'][sess], paths['glm_config'][sess],
                    fit_algo)
       
        #step 6. Compute Contrasts
        print "Computing contrasts"
        if side=='False':
            glm_tools.compute_contrasts(
                contrast, misc, CompletePaths, glms, model=model_id)
        else:
            cortical_glm.compute_contrasts(
                contrast, misc, CompletePaths, glms, model=model_id)
            
