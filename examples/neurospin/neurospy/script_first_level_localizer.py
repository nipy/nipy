"""
Script that perform the first-level analysis of a dataset of the FIAC
Last updated by B.Thirion

Author : Lise Favre, Bertrand Thirion, 2008-2010
"""
import os
from configobj import ConfigObj
from nipy.neurospin.utils.mask import compute_mask_files
import GLMTools, Contrast

# -----------------------------------------------------------
# --------- Set the paths -----------------------------------
#-----------------------------------------------------------

DBPath = "/volatile/thirion/Localizer"
Subjects = ["s12069"]#["s12277"]#, "s12300","s12401","s12431","s12508","s12532","s12635","s12636","s12826","s12898","s12913","s12919","s12920"]#["s12069"]#
Acquisitions = ["acquisition"]
Sessions = ["loc1"]
modelDir = "default"
fmri_wc = "S*.nii"

# ---------------------------------------------------------
# -------- General Information ----------------------------
# ---------------------------------------------------------

TR = 2.4
nbFrames = 128

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
hrfType = "Canonical With Derivative"

# Possible choices for drift : "Blank", "Cosine", "Polynomial"
drift = "Cosine"
cos_FreqCut = 128

#-------------- GLM options
# Possible choices : "Kalman_AR1", "Kalman", "Ordinary Least Squares"
fit_algo = "Kalman_AR1"

#-------------- Contrast Options
# Possible choices : "Contrast Name" or "Contrast Number"
save_mode = "Contrast Name"




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

# fixme : all the structures (misc, design_patrices, contrasts or mask)
# should be passed as structures not through files


# Treat sequentially all subjects & acquisitions
for s in Subjects:
    print "Subject : %s" % s
    
    for a in Acquisitions:
        # step 1. set all the paths
        basePath = os.sep.join((DBPath, s, "fMRI", a))
        paths = GLMTools.generate_all_brainvisa_paths( basePath, Sessions, 
                                                        fmri_wc, modelDir)  
          
        misc = ConfigObj(paths['misc'])
        misc["sessions"] = Sessions
        misc["tasks"] = Conditions
        misc["mask_url"] = paths['mask']
        misc.write()

        # step 2. Create one design matrix for each session
        design_matrices = {}
        for sess in Sessions:
            design_matrices[sess] =\
               GLMTools.DesignMatrix( nbFrames, paths['paradigm'], paths['misc'], 
                                       TR, paths['dmtx'][sess], sess, 
                                       hrfType=hrfType, drift=drift,  
                                       cos_FreqCut=cos_FreqCut, model=modelDir)        
        # step 3. Compute the Mask
        # fixme : it should be possible to provide a pre-computed mask
        print "Computing the Mask"
        mask_array = compute_mask_files( paths['fmri'].values()[0][0], 
                                         paths['mask'], True, infTh, supTh)
        
        # step 4. Creating functional contrasts
        print "Creating Contrasts"
        clist = Contrast.ContrastList(misc=misc)
        generate_localizer_contrasts(clist)
        contrast = clist.save_dic(paths['contrast_file'])
        CompletePaths = GLMTools.generate_brainvisa_ouput_paths( 
                        paths["contrasts"],  contrast)

        # step 5. Fit the  glm for each session
        glms = {}
        for sess in Sessions:
            print "Fitting GLM for session : %s" % sess
            glms[sess] = GLMTools.GLMFit(
                paths['fmri'][sess], design_matrices[sess],
                paths['glm_dump'][sess], paths['glm_config'][sess],
                fit_algo, paths['mask'])
            
        #step 6. Compute Contrasts
        print "Computing contrasts"
        GLMTools.ComputeContrasts(contrast, misc, glms, save_mode,
                                  CompletePaths=CompletePaths,
                                  threshold=3.0,
                                  cluster=10,
                                  method='None')

        
