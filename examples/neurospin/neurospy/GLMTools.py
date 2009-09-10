"""
General tools to analyse fMRI datasets (GLM fit)
using nipy.neurospin tools

Author : Lise Favre, Bertrand Thirion, 2008-2009
"""

import numpy as np
import commands
import os

from configobj import ConfigObj

from nipy.io.imageformats import load, save, Nifti1Image 


import Results
from nipy.neurospin.utils.mask import compute_mask_files

# ----------------------------------------------
# -------- IO functions ------------------------
# ----------------------------------------------

def load_image(image_path, mask_path=None ):
    """ Return an array of image data masked by mask data 

    Parameters
    ----------
    image_path string or list of strings that represent the data of interest
    mask_path=None: string that yields the mask path

    Returns
    -------
    image_data a data array that can be 1, 2, 3  or 4D 
               depending on chether mask==None or not
               and on the length of the times series
    """
    if mask_path !=None:
       rmask = load(mask_path)
       mask = rmask.get_data()
    else:
        mask = None

    image_data = []
    
    if hasattr(image_path, '__iter__'):
       if len(image_path)==1:
          image_path = image_path[0]

    if hasattr(image_path, '__iter__'):
       for im in image_path:
           temp = load(im).get_data()     
           if mask != None:
               temp = load(im).get_data()[mask,:] 
           else:
                temp = load(im).get_data()  
           image_data.append(temp)
    else:
         image_data = load(image_path).get_data()
         if mask != None:
              image_data = image_data[mask>0,:]
                
    return image_data


def save_volume(shape, path, affine, mask=None, data=None, descrip=None):
    """
    niftilib-based save volume utility for masked volumes
    
    Parameters
    ----------
    shape, tupe of dimensions of the data
    path image path
    affine, transformation of the grid to a coordinate system
    mask=None, binary mask used to reduce th colume variable
    data=None data to be put in the volume
    descrip=None, a string descibing what the image is
    """
    volume = np.zeros(shape)
    if mask== None: 
       print "Could not write the image: no data"
       return

    if data == None:
       print "Could not write the image:no mask"
       return

    if np.size(data.shape) == 1:
        volume[mask > 0] = data
    else:
        for i in range(data.shape[0]):
            volume[i][mask[0] > 0] = data[i]

    wim = Nifti1Image(volume, affine)
    if descrip !=None:
        wim.get_header()['descrip']=descrip
    save(wim, path)

def save_all(contrast, ContrastId, dim, mask_url, kargs):
    """
    Save all the outputs of a GLM analysis + contrast definition
    
    Parameters
    ----------
    contrast a structure describing the values related to the computed contrast 
    ContrastId, string, the contrast identifier
    dim the dimension of the contrast
    mask_url path of the mask image related to the data
    kargs, should have the key 'paths', 
           that yield the paths where everything should be written 
           optionally it can also have the keys 'method', 'threshold' and 'cluster'
           that are used to define the parameters for vizualization of the html page. 
        
    fixme : handle the case mask=None
    """
    # prepare the paths (?)
    if kargs.has_key("paths"):
        paths = kargs["paths"]
    else:
        print "Cannot save contrast files. Missing argument : paths"
        return
    
    mask = load(mask_url)
    mask_arr = mask.get_data()
    affine = mask.get_affine()
    shape = mask.get_shape()    
    contrasts_path = paths["Contrasts_path"]
    
    t = contrast.stat()
    z = contrast.zscore()

    # saving the Z statsitics map
    results = "Z map"
    z_file = os.sep.join((contrasts_path, "%s_%s.nii"% (str(ContrastId),
                                                        paths[results])))
    save_volume(shape, z_file, affine, mask_arr, z, results)
    
    # Saving the t/F statistics map
    if contrast.type == "t":
        results = "Student-t tests"
    elif contrast.type == "F":
        results = "Fisher tests"
    t_file = os.sep.join((contrasts_path, "%s_%s.nii" %
                          (str(ContrastId), paths[results])))
    save_volume(shape, t_file, affine, mask_arr, t, results)
    
    if int(dim) != 1:
        shape = (int(dim) * int(dim), shape[0], shape[1], shape[2])
        contrast.variance = contrast.variance.reshape(int(dim)**2, -1)

    ## saving the associated variance map
    # fixme : breaks with F contrasts !
    if contrast.type == "t":
        results = "Residual variance"
        res_file = os.sep.join((contrasts_path, "%s_%s.nii" %
                            (str(ContrastId), paths[results])))
        save_volume(shape, res_file, affine, mask_arr,
                    contrast.variance)
        if int(dim) != 1:
            shape = (int(dim), shape[1], shape[2], shape[3])

    # writing the associated contrast structure
    # fixme : breaks with F contrasts !
    if contrast.type == "t":
        results = "contrast definition"
        con_file = os.sep.join((contrasts_path, "%s_%s.nii" %
                                (str(ContrastId), paths[results])))
        save_volume(shape, con_file, affine, mask_arr,
                    contrast.effect)
 
    # writing the results as an html page
    if kargs.has_key("method"):
        method = kargs["method"]
    else:
        method = 'fpr'

    if kargs.has_key("threshold"):
        threshold = kargs["threshold"]
    else:
        threshold = 0.001

    if kargs.has_key("cluster"):
        cluster = kargs["cluster"]
    else:
        cluster = 0
 
    html_file = os.sep.join((contrasts_path, "%s_%s.nii" % (str(ContrastId), 
              paths[results])))
  
    Results.ComputeResultsContents(z_file, mask_url, html_file,
                                   threshold=threshold, method=method,
                                   cluster=cluster)


def ComputeMask(fmriFiles, outputFile, infT=0.4, supT=0.9):
    """
    Perform the mask computation, see the api of  compute_mask_files
    """ 
    compute_mask_files( fmriFiles, outputFile, False, infT, supT, cc=1)



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
    
    Parameters
    ----------
    nbFrames
    paradigm
    miscFile
    tr
    outputFile:
    session
    DmtxParam
    
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
    import DesignMatrix as dm
   
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
    
    _design = design.compute_design(session,verbose=1)
    
    if _design != None:
        design.save_csv(outputFile)

def GLMFit(file, designMatrix,  output_glm, outputCon,
           fit="Kalman_AR1", mask_url=None):
    """
    Call the GLM Fit function with apropriate arguments

    Parameters
    ----------
    file, string or list of strings,
          path of the fMRI data file(s)
    designmatrix, string, path of the design matrix .csv file 
    mask_url=None string, path of the mask file
          if None, no mask is applied
    output_glm, string, 
                path of the output glm .npz dump
    outputCon, string,
               path of the output configobj contrast object
    fit= 'Kalman_AR1', string to be chosen among
         "Kalman_AR1", "Ordinary Least Squares", "Kalman"
         that represents both the model and the fit method
                
    Returns
    -------
    glm, a nipy.neurospin.glm.glm instance representing the GLM

    fixme: mask should be optional
    """
    if fit == "Kalman_AR1":
        model = "ar1"
        method = "kalman"
    elif fit == "Ordinary Least Squares":
        method = "ols"
        model="spherical"
    elif fit == "Kalman":
        method = "kalman"
        model = "spherical"
    
    
    import DesignMatrix as dm
    names, X = dm.load_dmtx_from_csv(designMatrix)
      
    Y = load_image(file, mask_url)

    import nipy.neurospin.glm as GLM
    glm = GLM.glm()
    glm.fit(Y.T, X, method=method, model=model)
    glm.save(output_glm)      
    cobj = ConfigObj(outputCon)
    cobj["DesignFilePath"] = designMatrix
    cobj["mask_url"] = mask_url
    cobj.write()   
   
    return glm

def ComputeContrasts(contrastFile, miscFile, glms, save_mode="Contrast Name", 
                                   model = "default", verbose=0, **kargs):
    """
    Contrast computation utility    
    
    Parameters
    ----------
    contrastFile, string, path of the configobj contrast file
    miscFile, string, path of the configobj miscinfo file
    glms, list of nipy.neurospin.glm.glm instances 
          representing the glms of the individual sessions 
    save_mode="Contrast Name", string to be chosen
                        among "Contrast Name" or "Contrast Number"
    model="default", string,
                     name of the contrast model used in miscfile
    """
    import nipy.neurospin.glm as GLM
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
        else:
            raise ValueError, "unknown save mode"

        for key, value in contrasts[contrast].items():
            if verbose: print key,value
            if key != "Type" and key != "Dimension":
                session = "_".join(key.split("_")[:-1])
                if not designs.has_key(session):
                    print "Loading session : %s" % session
                designs[session] = GLM.load(glms[session]["GlmDumpFile"])

                bv=[int(j) != 0 for j in value]
                if contrast_type == "t" and sum(bv)>0:
                    _con = designs[session].contrast([int(i) for i in value])
                    final_contrast.append(_con)

                if contrast_type == "F":
                    if not multicon.has_key(session):
                        multicon[session] = np.array(bv)
                    else:
                        multicon[session] = np.vstack((multicon[session], bv))
        if contrast_type == "F":
            for key, value in multicon.items():
                if sum([j != 0 for j in value.reshape(-1)]) != 0:
                    _con = designs[key].contrast(value)
                    final_contrast.append(_con)

        design = designs[session]
        res_contrast = final_contrast[0]
        for c in final_contrast[1:]:
            res_contrast = res_contrast + c
            res_contrast.type = contrast_type
            
        mask_url = None
        if misc.has_key("mask"):
           mask_url = misc["mask"]
        save_all(res_contrast, ContrastId, contrast_dimension, mask_url, kargs)
        misc[model]["con_dofs"][contrast] = res_contrast.dof
    misc["Contrast Save Mode"] = save_mode
    misc.write()


# -----------------------------------------------------------
# --- functions that depend on VBA --------------------------
# -----------------------------------------------------------

def GLMFit_(file, designMatrix, mask, outputVBA, outputCon,
           fit="Kalman_AR1"):
    """
    Call the GLM Fit function with apropriate arguments

    Parameters
    ----------
    file
    designmatrix
    mask
    outputVBA
    outputCon
    fit='Kalman_AR1'
    
    Returns
    -------
    glm, a vba.VBA instance representing the GLM
    
    """
    from vba import VBA
    from dataFrame import DF
    if fit == "Kalman_AR1":
        model = "ar1"
        method = "kalman"
    elif fit == "Ordinary Least Squares":
        method = "ols"
        model="spherical"
    elif fit == "Kalman":
        method = "kalman"
        model = "spherical"
    
    s = dict()
    s["GlmDumpFile"] = outputVBA
    s["ConfigFilePath"] = outputCon
    s["DesignFilePath"] = designMatrix
       
    
    tab = DF.read(designMatrix)
        
    glm = VBA(tab, mask_url=mask, create_design_mat = False, mri_names = file, 
                   model = model, method = method)
    glm.fit()
    glm.save(s)
    return glm

def ComputeContrasts_(contrastFile, miscFile, glms, save_mode="Contrast Name", 
                                   model = "default", **kargs):
    """
    Contrast computation utility    
    
    Parameters
    ----------
    contrastFile
    miscFile
    glms
    save_mode="Contrast Name"
    model="default"
    """
    from vba import VBA
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

                bv=[int(j) != 0 for j in value]
                if contrast_type == "t" and sum(bv)>0:
                    designs[session].contrast([int(i) for i in value])
                    final_contrast.append(designs[session]._con)

                if contrast_type == "F":
                    if not multicon.has_key(session):
                        multicon[session] = np.array(bv)
                    else:
                        multicon[session] = np.vstack((multicon[session], bv))
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

        
        save_all(res_contrast, ContrastId, contrast_dimension,
                 design.mask_url, kargs)
        misc[model]["con_dofs"][contrast] = res_contrast.dof
    misc["Contrast Save Mode"] = save_mode
    misc.write()
