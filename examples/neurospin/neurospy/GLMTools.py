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
from nipy.neurospin.utils.mask import compute_mask_files

import Results


# ----------------------------------------------
# -------- IO functions ------------------------
# ----------------------------------------------


def load_image(image_path, mask_path=None ):
    """ Return an array of image data masked by mask data 

    Parameters
    ----------
    image_path string or list of strings 
               that yields the data of interest
    mask_path=None: string that yields the mask path

    Returns
    -------
    image_data a data array that can be 1, 2, 3  or 4D 
               depending on chether mask==None or not
               and on the length of the times series
    """
    # fixme : do some check
    if mask_path !=None:
       rmask = load(mask_path)
       shape = rmask.get_shape()[:3]
       mask = np.reshape(rmask.get_data(),shape)
    else:
        mask = None

    image_data = []
    
    if hasattr(image_path, '__iter__'):
       if len(image_path)==1:
          image_path = image_path[0]

    if hasattr(image_path, '__iter__'):
       for im in image_path:
           if mask != None:
               temp = np.reshape(load(im).get_data(),shape)[mask>0,:]
               
           else:
                temp = np.reshape(load(im).get_data(),shape) 
           image_data.append(temp)
       image_data = np.array(image_data).T
    else:
         image_data = load(image_path).get_data()
         if mask != None:
              image_data = image_data[mask>0,:]
    
    return image_data

def save_masked_volume(data, mask_url, path, descrip=None):
    """
    volume saving utility for masked volumes
    
    Parameters
    ----------
    data, array of shape(nvox) data to be put in the volume
    mask_url, string, the mask path
    path string, output image path
    descrip = None, a string descibing what the image is
    """
    rmask = load(mask_url)
    mask = rmask.get_data()
    shape = rmask.get_shape()
    affine = rmask.get_affine()
    save_volume(shape, path, affine, mask, data, descrip)
    

def save_volume(shape, path, affine, mask=None, data=None, descrip=None):
    """
    volume saving utility for masked volumes
    
    Parameters
    ----------
    shape, tupe of dimensions of the data
    path, string, output image path
    affine, transformation of the grid to a coordinate system
    mask=None, binary mask used to reduce the volume size
    data=None data to be put in the volume
    descrip=None, a string descibing what the image is

    Fixme
    -----
    Handle the case where data is multi-dimensional
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

def save_all_images(contrast, dim, mask_url, kargs):
    """
    idem savel_all, but the names are now all included in kargs
    """
    z_file = kargs["z_file"]
    t_file = kargs["t_file"]
    res_file = kargs["res_file"]
    con_file = kargs["con_file"]
    html_file = kargs["html_file"]
    mask = load(mask_url)
    mask_arr = mask.get_data()
    affine = mask.get_affine()
    shape = mask.get_shape()    
   
    # load the values
    t = contrast.stat()
    z = contrast.zscore()

    # saving the Z statistics map
    save_volume(shape, z_file, affine, mask_arr, z, "z_file")
    
    # Saving the t/F statistics map
    save_volume(shape, t_file, affine, mask_arr, t, "t_file")
    
    if int(dim) != 1:
        shape = (shape[0], shape[1], shape[2],int(dim)**2)
        contrast.variance = contrast.variance.reshape(int(dim)**2, -1)

    ## saving the associated variance map
    # fixme : breaks with F contrasts !
    if contrast.type == "t":
        save_volume(shape, res_file, affine, mask_arr,
                    contrast.variance)
    if int(dim) != 1:
        shape = (shape[0], shape[1], shape[2], int(dim))

    # writing the associated contrast structure
    # fixme : breaks with F contrasts !
    if contrast.type == "t":
        save_volume(shape, con_file, affine, mask_arr, contrast.effect)
         
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

    Results.ComputeResultsContents(z_file, mask_url, html_file,
                                   threshold=threshold, method=method,
                                   cluster=cluster)


def save_all(contrast, ContrastId, dim, mask_url, kargs):
    """
    Save all the images related to one contrast
    
    Parameters
    ----------
    contrast a structure describing 
             the values related to the computed contrast 
    ContrastId, string, the contrast identifier
    dim the dimension of the contrast
    mask_url path of the mask image related to the data
    kargs, should have the key 'paths', 
           that yield the paths where everything should be written 
           optionally it can also have the keys 
           'method', 'threshold' and 'cluster'
           that are used to define the parameters for vizualization 
           of the html page. 
        
    fixme : handle the case mask=None
    """
    
    # prepare the paths
    if kargs.has_key("paths"):
        paths = kargs["paths"]
    else:
        print "Cannot save contrast files. Missing argument : paths"
        return
    contrasts_path = paths["Contrasts_path"]
    results = "Z map"
    z_file = os.sep.join((contrasts_path, "%s_%s.nii"% (str(ContrastId),
                                                        paths[results])))
    if contrast.type == "t":
        results = "Student-t tests"
    elif contrast.type == "F":
        results = "Fisher tests"
    t_file = os.sep.join((contrasts_path, "%s_%s.nii" %
                          (str(ContrastId), paths[results])))
    results = "Residual variance"
    res_file = os.sep.join((contrasts_path, "%s_%s.nii" %
        (str(ContrastId), paths[results])))
    results = "contrast definition"
    con_file = os.sep.join((contrasts_path, "%s_%s.nii" %
        (str(ContrastId), paths[results])))
    results="HTML results"
    html_file = os.sep.join((contrasts_path, "%s_%s.html" % (str(ContrastId), 
              paths[results])))
    kargs["z_file"] = z_file
    kargs["t_file"] = t_file
    kargs["res_file"] = res_file
    kargs["con_file"] = con_file
    kargs["html_file"] = html_file

    save_all_images(contrast, dim, mask_url, kargs)
 

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
    FIR_delays = DmtxParam["FIR_delays"]
    FIR_duration  = DmtxParam["FIR_duration"]
    
    model = 'default'

    # set RegMatrix
    #regMatrix = DmtxParam["drift_matrix"]
    regMatrix = None 
    if DmtxParam.has_key('reg_matrix'):
        regMatrix = DmtxParam["reg_matrix"]

    # set RegNames
    regNames = None
    if DmtxParam.has_key('reg_names'):
        regNames = DmtxParam["reg_names"]

    _DesignMatrix(nbFrames, paradigm, miscFile, tr, outputFile,
                  session, hrfType, drift, regMatrix, poly_order,
                  cos_FreqCut, FIR_delays, FIR_duration, model, regNames)



def _DesignMatrix(nbFrames, paradigm, miscFile, tr, outputFile,
                  session, hrfType="Canonical", drift="Blank",
                  regMatrix=None, poly_order=2, cos_FreqCut=128,
                  FIR_delays=[0], FIR_duration=1., model="default",
                  regNames=None, verbose=0):
    """
    Base function to define design matrices
    fixme : control that the FIR model works
    """
    import DesignMatrix as dm
   
    design = dm.DesignMatrix(nbFrames, paradigm, session, miscFile, model)
    design.load()
    design.timing(tr)
 
    # fixme : append had-defined regressors (e.g. motion)
    # set the drift terms
    design.set_drift(drift,  poly_order, cos_FreqCut)

    # set the condition-related regressors
    design.set_conditions(hrfType, FIR_delays, FIR_duration)

    _design = design.compute_design(regMatrix, regNames, session, verbose)
    
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
                                   model="default", verbose=0, **kargs):
    """
    Contrast computation utility    
    
    Parameters
    ----------
    contrastFile, string, path of the configobj contrast file
    miscFile, string, path of the configobj miscinfo file
    glms, dictionary indexed by sessions that yields paths to
          nipy.neurospin.glm.glm instances 
          representing the glms of the individual sessions 
          in particular , it should have the key "GlmDumpFile".
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
        k = i+1
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
                if verbose and (not designs.has_key(session)):
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

        if contrasts.has_key('mask_url'):
            mask_url = contrasts['mask_url']
        elif misc.has_key('mask_url'):
            mask_url = misc['mask_url']
        else:
            mask_url = None

        if kargs.has_key('CompletePaths'):
           # just write the results at the provided paths, 
           # assuming they are coorect
            cpp = kargs['CompletePaths'][contrast]
            for k in cpp.keys():
                kargs[k] = cpp[k]
            save_all_images(res_contrast, contrast_dimension, mask_url, kargs)
                                          
        else:
            # partially recompute the paths
            save_all(res_contrast, ContrastId, contrast_dimension, mask_url, 
                     kargs)
        misc[model]["con_dofs"][contrast] = res_contrast.dof
    misc["Contrast Save Mode"] = save_mode
    misc.write()


