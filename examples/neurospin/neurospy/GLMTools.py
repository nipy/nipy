"""
General tools to analyse fMRI datasets (FSL pre-rocessing and GLM fit)
using nipy.neurospin tools

Author : Lise Favre, Bertrand Thirion, 2008-2009
"""

import numpy as np
from numpy import *
import commands
import nifti
import os

from configobj import ConfigObj
import scipy.ndimage as sn

from vba import VBA
import Results
from nipy.neurospin.utils.mask import compute_mask_files

# ----------------------------------------------
# -------- Ancillary functions -----------------
# ----------------------------------------------

def save_volume(volume, file, header, mask=None, data=None):
    """
    niftilib-based save volume utility

    fixme :  very low-level and naive 
    """
    if mask != None and data != None:
        if size(data.shape) == 1:
            volume[mask > 0] = data
        else:
            for i in range(data.shape[0]):
                volume[i][mask[0] > 0] = data[i]
        nifti.NiftiImage(volume,header).save(file)

def saveall(contrast, design, ContrastId, dim, kargs):
    """
    Save all the outputs of a GLM analysis + contrast definition
    fixme : restructure it
    """
    # preparae the paths (?)
    if kargs.has_key("paths"):
        paths = kargs["paths"]
    else:
        print "Cannot save contrast files. Missing argument : paths"
        return
    mask = nifti.NiftiImage(design.mask_url)
    mask_arr = mask.asarray()
    header = mask.header
    contrasts_path = paths["Contrasts_path"]
    if size(mask_arr.shape) == 3:
        mask_arr= mask_arr.reshape(1, mask_arr.shape[0],
                                   mask_arr.shape[1], mask_arr.shape[2])
    shape = mask_arr.shape
    t = contrast.stat()
    z = contrast.zscore()

    # saving the Z statsitics map
    results = "Z map"
    z_file = os.sep.join((contrasts_path, "%s_%s.nii"% (str(ContrastId), paths[results])))
    save_volume(zeros(shape), z_file, header, mask_arr, z)
    
    # Saving the t/F statistics map
    if contrast.type == "t":
        results = "Student-t tests"
    elif contrast.type == "F":
        results = "Fisher tests"
    t_file = os.sep.join((contrasts_path, "%s_%s.nii" %
                          (str(ContrastId), paths[results])))
    save_volume(zeros(shape), t_file, header, mask_arr, t)
    if int(dim) != 1:
        shape = (int(dim) * int(dim), shape[1], shape[2], shape[3])
        contrast.variance = contrast.variance.reshape(int(dim) * int(dim), -1)

    ## saving the associated variance map
    # fixme : breaks with F contrasts !
    if contrast.type == "t":
        results = "Residual variance"
        res_file = os.sep.join((contrasts_path, "%s_%s.nii" %
                            (str(ContrastId), paths[results])))
        save_volume(zeros(shape), res_file, header, mask_arr, contrast.variance)
        if int(dim) != 1:
            shape = (int(dim), shape[1], shape[2], shape[3])

    # writing the associated contrast structure
     # fixme : breaks with F contrasts !
    if contrast.type == "t":
        results = "contrast definition"
        con_file = os.sep.join((contrasts_path, "%s_%s.nii" %
                                (str(ContrastId), paths[results])))
        save_volume(zeros(shape), con_file, header, mask_arr, contrast.effect)

    # writing the results as an html page
    if kargs.has_key("method"):
        method = kargs["method"]
    else:
        method = 'fpr'
        #print "Cannot save HTML results. Missing argument : method"
        #return

    if kargs.has_key("threshold"):
        threshold = kargs["threshold"]
    else:
        threshold=0.001
        #print "Cannot save HTML results. Missing argument : threshold"
        #return

    if kargs.has_key("cluster"):
        cluster = kargs["cluster"]
    else:
        cluster = 0

    results = "HTML Results"
    html_file = os.sep.join((contrasts_path,
                             "%s_%s.html" % (str(ContrastId),
                                             paths[results])))
    Results.ComputeResultsContents(z_file, design.mask_url, html_file,
                                   threshold=threshold, method=method,
                                   cluster=cluster)


def ComputeMask(fmriFiles, outputFile, infT=0.4, supT=0.9):
    """
    Perform the mask computation
    """
    compute_mask_files( fmriFiles, outputFile, False, infT, supT, cc=1)

# ---------------------------------------------
# various FSL-based Pre processings functions -
# ---------------------------------------------

def SliceTiming(file, tr, outputFile, interleaved = False, ascending = True):
    """
    Perform slice timing using FSL
    """
    so = " "
    inter = " "
    if interleaved:
        inter = "--odd"
    if not ascending:
        so = "--down"
    print "slicetimer -i '%s' -o '%s' %s %s -r %s" % (file, outputFile, so, inter, str(tr))
    print commands.getoutput("slicetimer -i '%s' -o '%s' %s %s -r %s" % (file, outputFile, so, inter, str(tr)))

def Realign(file, refFile, outputFile):
    """
    Perform realignment using FSL
    """
    print commands.getoutput("mcflirt -in '%s' -out '%s' -reffile '%s' -mats" % (file, outputFile, refFile))

def NormalizeAnat(anat, templatet1, normAnat, norm_matrix, searcht1 = "NASO"):
    """
    Form the normalization of anatomical images using FSL
    """
    if searcht1 == "AVA":
        s1 = "-searchrx -0 0 -searchry -0 0 -searchrz -0 0"
    elif (searcht1 == "NASO"):
        s1 = "-searchrx -90 90 -searchry -90 90 -searchrz -90 90"
    elif (searcht1 == "IO"):
        s1 = "-searchrx -180 180 -searchry -180 180 -searchrz -180 180"
    print "T1 MRI on Template\n"
    print commands.getoutput("flirt -in '%s' -ref '%s' -omat '%s' -out '%s' -bins 1024 -cost corratio %s -dof 12" % (anat, templatet1, norm_matrix, normAnat, s1) )
    print "Finished"

def NormalizeFMRI(file, anat, outputFile, normAnat, norm_matrix, searchfmri = "AVA"):
    """
    Perform the normalization of fMRI data using FSL
    """
    if searchfmri == "AVA":
        s2 = "-searchrx -0 0 -searchry -0 0 -searchrz -0 0"
    elif (searchfmri == "NASO"):
        s2 = "-searchrx -90 90 -searchry -90 90 -searchrz -90 90"
    elif (searchfmri == "IO"):
        s2 = "-searchrx -180 180 -searchry -180 180 -searchrz -180 180"
    print "fMRI on T1 MRI\n"
    print commands.getoutput("flirt -in '%s' -ref '%s' -omat /tmp/fmri1.mat -bins 1024 -cost corratio %s -dof 6" % (file, anat, s2))
    print "fMRI on Template\n"
    print commands.getoutput("convert_xfm -omat /tmp/fmri.mat -concat '%s' /tmp/fmri1.mat" % norm_matrix)
    print commands.getoutput("flirt -in '%s' -ref '%s' -out '%s' -applyxfm -init /tmp/fmri.mat -interp trilinear" % (file, normAnat, outputFile))
    print "Finished\n"

def Smooth(file, outputFile, fwhm):
    """
    fixme : this might smooth each slice indepently ?
    """
    #  voxel_width = 3
    fmri = nifti.NiftiImage(file)
    #voxel_width = fmri.header['voxel_size'][2]
    voxel_width = fmri.header['pixdim'][2]
    sigma = fwhm/(voxel_width*2*sqrt(2*log(2)))
    for i in fmri.data:
        sn.gaussian_filter(i, sigma, order=0, output=None,
                           mode='reflect', cval=0.0)
    fmri.save(outputFile)


#-----------------------------------------------------
#------- First Level analysis ------------------------
#-----------------------------------------------------

def CheckDmtxParam(DmtxParam):
    """
    check that Dmtx parameters are OK
    """
    pass

def DesignMatrix(nbFrames, paradigm, miscFile, tr, outputFile,
                 session, DmtxParam):
    """
    Higher level function to define design matrices
    This function simply unfolds  the Dmtxparam dictionary
    and calls the _DesignMatrix function
    
    Parameters:
    -----------
    - nbFrames
    - paradigm
    - miscFile
    - tr
    - outputFile:
    - session
    - DmtxParam
    
    """
    hrfType = DmtxParam["hrfType"]
    drift = DmtxParam["drift"]
    poly_order = DmtxParam["poly_order"]
    cos_FreqCut = DmtxParam["cos_FreqCut"] 
    FIR_order = DmtxParam["FIR_order"]
    FIR_length  = DmtxParam["FIR_length"]
    driftMatrix = DmtxParam["drift_matrix"]
    model = 'default'# this is a brainvisa thing for misc info 
    _DesignMatrix(nbFrames, paradigm, miscFile, tr, outputFile,
                  session, hrfType, drift, driftMatrix, poly_order,
                  cos_FreqCut, FIR_order, FIR_length, model)



def _DesignMatrix(nbFrames, paradigm, miscFile, tr, outputFile,
                  session, hrfType="Canonical", drift="Blank",
                  driftMatrix=None, poly_order=2, cos_FreqCut=128,
                  FIR_order=1, FIR_length=1, model="default", verbose=0):
    """
    Base function to define design matrices
    """
    from nipy.modalities.fmri import formula, utils, hrf
    
    ## For DesignMatrix
    import DesignMatrix as dm
    from dataFrame import DF
    
    design = dm.DesignMatrix(nbFrames, paradigm, session, miscFile, model)
    design.load()
    design.timing(tr)

    
    """
    fixme : set the FIR model
    # set the hrf
    if hrf == "Canonical":
        hrf = dm.hrf.glover
    elif hrf == "Canonical With Derivative":
        hrf = dm.hrf.glover_deriv
    elif hrf == "FIR Model":
        design.compute_fir_design(drift = pdrift, name = session,
                                  o = FIR_order, l = FIR_length)
        output = DF(colnames=design.names, data=design._design)
        output.write(outputFile)
        return 0
    else:
        print "Not HRF model passed. Aborting process."
        return
    """
        
    # fixme : append had-defined regressors (e.g. motion)
    # set the drift terms
    design.set_drift(drift,  poly_order, cos_FreqCut)

    # set the condition-related regressors
    # fixme : set the FIR model
    design.set_conditions(hrfType)
    
    design.compute_design(session,verbose=1)
    
    if hasattr(design, "names"):
        output = DF(colnames=design.names, data=design._design)
        if verbose : print design.names
        output.write(outputFile)
    

def GLMFit(file, designMatrix, mask, outputVBA, outputCon,
           fit="Kalman_AR1"):
    """
    Call the GLM Fit function with apropriate arguments

    Parameters:
    -----------
    - file
    - designmatrix
    - mask
    - outputVBA
    - outputCon
    - fit='Kalman_AR1'
    
    Output:
    -------
    - glm
    
    """
    from dataFrame import DF
    tab = DF.read(designMatrix)
    
    if fit == "Kalman_AR1":
        model = "ar1"
        method = "kalman"
    elif fit == "Ordinary Least Squares":
        method = "ols"
        model="spherical"
    elif fit == "Kalman":
        method = "kalman"
        model = "spherical"

    glm = VBA(tab, mask_url=mask, create_design_mat = False, mri_names = file, model = model, method = method)
    glm.fit()
    s=dict()
    s["GlmDumpFile"] = outputVBA
    s["ConfigFilePath"] = outputCon
    s["DesignFilePath"] = designMatrix
    glm.save(s)
    return glm


def ComputeContrasts(contrastFile, miscFile, glms, save_mode="Contrast Name",
                     model = "default", **kargs):
    """
    """
    verbose = 0 # fixme: put ine the kwargs
    misc = ConfigObj(miscFile)
    if not misc.has_key(model):
        misc[model] = {}

    if not misc[model].has_key("con_dofs"):
        misc[model]["con_dofs"] = {}

    contrasts = ConfigObj(contrastFile)
    contrasts_names = contrasts["contrast"]
    designs = {}
    for i, contrast in enumerate(contrasts_names):
        contrast_type = contrasts[contrast]["Type"]
        contrast_dimension = contrasts[contrast]["Dimension"]
        final_contrast = []
        k = i + 1
        multicon = dict()
        if save_mode == "Contrast Name":
            ContrastId = contrast
        elif save_mode == "Contrast Number":
            ContrastId = "%04i" % k
            
        for key, value in contrasts[contrast].items():
            if verbose: print key,value
            if key != "Type" and key != "Dimension":
                session = "_".join(key.split("_")[:-1])
                if not designs.has_key(session):
                    print "Loading session : %s" % session
                designs[session] = VBA(glms[session])

                if contrast_type == "t" and sum([int(j) != 0 for j in value]) != 0:
                    designs[session].contrast([int(i) for i in value])
                    final_contrast.append(designs[session]._con)

                if contrast_type == "F":
                    if not multicon.has_key(session):
                        multicon[session] = array([int(i) for i in value])
                    else:
                        multicon[session] = vstack((multicon[session], [int(i) for i in value]))
        if contrast_type == "F":
            for key, value in multicon.items():
                if sum([j != 0 for j in value.reshape(-1)]) != 0:
                    designs[key].contrast(value)
                    final_contrast.append(designs[key]._con)

        design = designs[session]
        res_contrast = final_contrast[0]
        for c in final_contrast[1:]:
            res_contrast = res_contrast + c
            res_contrast.type = contrast_type
        saveall(res_contrast, design, ContrastId, contrast_dimension, kargs)
        misc[model]["con_dofs"][contrast] = res_contrast.dof
    misc["Contrast Save Mode"] = save_mode
    misc.write()
