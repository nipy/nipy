# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Script that perform the first-level analysis of a dataset of the localizer
Here the analysis is perfomed on the cortical of one hemisphere.


Author : Lise Favre, Bertrand Thirion, 2008-2010
"""
import os

from numpy import arange

from nipy.externals.configobj import ConfigObj

from nipy.neurospin.glm_files_layout import (glm_tools,
                                             contrast_tools,
                                             cortical_glm)

# -----------------------------------------------------------
# --------- Set the paths -----------------------------------
#-----------------------------------------------------------

DBPath = "/neurospin/lnao/Pmad/alan/subjfreesurfer"
Subjects = ['s12069']#['s12300', 's12401', 's12431', 's12508', 's12532', 's12539', 's12562','s12590', 's12635', 's12636', 's12898', 's12913', 's12919', 's12920']
Acquisitions = [""]
Sessions = ["loc1"]
model_id = "default"
side = 'right'
fmri_wc = "rh*.tex"
if side=='left':
    fmri_wc = "lh*.tex"


# ---------------------------------------------------------
# -------- General Information ----------------------------
# ---------------------------------------------------------

tr = 2.4
nb_frames = 128
frametimes = arange(nb_frames) * tr

Conditions = [ 'damier_H', 'damier_V', 'clicDaudio', 'clicGaudio', 
'clicDvideo', 'clicGvideo', 'calculaudio', 'calculvideo', 'phrasevideo', 
'phraseaudio' ]


# ---------------------------------------------------------
# ------ First level analysis parameters ---------------------
# ---------------------------------------------------------

#---------- Masking parameters 
infTh = 0.4
supTh = 0.9

#---------- Design Matrix

# Possible choices for hrfType : "Canonical", \
# "Canonical With Derivative" or "FIR"
hrf_model = "Canonical With Derivative"

# Possible choices for drift : "Blank", "Cosine", "Polynomial"
drift_model = "Cosine"
hfcut = 128

#-------------- GLM options
# Possible choices : "Kalman_AR1", "Kalman", "Ordinary Least Squares"
fit_algo = "Kalman_AR1"


def generate_localizer_contrasts(contrast):
    """
    This utility appends standard localizer contrasts
    to the input contrast structure

    Parameters
    ----------
    contrast: configObj
        that contains the automatically generated contarsts

    Caveat
    ------
    contrast is changed in place
    """
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
    
# ------------------------------------------------------------------
# Launching Pipeline on all subjects, all acquisitions, all sessions 
# -------------------------------------------------------------------

# Treat sequentially all subjects & acquisitions
for s in Subjects:
    print "Subject : %s" % s
    
    for a in Acquisitions:
        # step 1. set all the paths
        basePath = os.sep.join((DBPath, s, "fct", a))
        paths = cortical_glm.generate_all_brainvisa_paths(
            basePath, Sessions, fmri_wc, model_id)  
          
        misc = ConfigObj(paths['misc'])
        misc["sessions"] = Sessions
        misc["tasks"] = Conditions
        misc.write()

        # step 2. Create one design matrix for each session
        design_matrices = {}
        for sess in Sessions:
            design_matrices[sess] = glm_tools.design_matrix(
                paths['misc'], paths['dmtx'][sess], sess, paths['paradigm'],
                frametimes, hrf_model=hrf_model, drift_model=drift_model,
                hfcut=hfcut, model=model_id)
                            
        # step 4. Creating functional contrasts
        print "Creating Contrasts"
        clist = contrast_tools.ContrastList(
            misc=ConfigObj(paths['misc']), model=model_id)
        generate_localizer_contrasts(clist)
        contrast = clist.save_dic(paths['contrast_file'])
        CompletePaths = cortical_glm.generate_brainvisa_ouput_paths( 
                        paths["contrasts"],  contrast, side)

        # step 5. Fit the  glm for each session
        glms = {}
        for sess in Sessions:
            print "Fitting GLM for session : %s" % sess
            glms[sess] = cortical_glm.glm_fit(
                paths['fmri'][sess], design_matrices[sess],
                paths['glm_dump'][sess], paths['glm_config'][sess], fit_algo)
            
        #step 6. Compute Contrasts
        print "Computing contrasts"
        cortical_glm.compute_contrasts(
            contrast, misc, CompletePaths, glms, model=model_id)

        
