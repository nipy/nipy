# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module contains several utiility functions to perform GLM 
on datasets that are organized according to the layout chosen in brainvisa.
It is thus assumed that: 
1. within a certain 'base_path' directory(*), 
   there are directories named with a certain session_id, 
   that contain the fMRI data ready to analyse
2. optionally, within the 'base_path' directory, 
   there is also a 'Minf' directory that contains a .csv file 
   that describes the paradigm used.
   
(*) the basepath directory is intended to correspond 
    to a session of acquistion of a given subjects. 
    It can contains multiple sessions.

Based on this architecture, the module conatins functionalities to
- estimate the design matrix
- load the data
- estimate the linear model
- estimate contrasts related to the linear model
- write output imges
- write an html page that summarizes the results 

Note that contrast specification relied on the  contrast_tools module

Author : Lise Favre, Bertrand Thirion, 2008-2010
"""

import os
import glob

import numpy as np

from ...externals.configobj import ConfigObj

from nibabel import load, save, Nifti1Image

############################################
# Path definition
############################################

def generate_all_brainvisa_paths( base_path, sessions, fmri_wc,  model_id,
                                  misc_id="misc_info.con", mask_id="mask.nii", 
                                  paradigm_id="paradigm.csv", 
                                  contrast_id="contrast.con",
                                  design_id="design_mat.csv",
                                  glm_dump="vba.npz",
                                  glm_config="vba_config.con"):
    """
    This function returns a dictionary with all the paths of all the paths
    where something id read or written in a standard GLM.
    The hard-coded paths reflect brainvisa database conventions.
    Additionally, this creates the missing output directories.

    Parameters
    ----------
    base_path: string,
              path of the acquition 
              (contains database, subject and acquitision ids)
    sessions: list of strings
              list of all the session related to the acquisition
    fmri_wc: string,
             wildcard for fMRI data files,
             assumed to be the same for all sessions
    model_id: string,
              identifier of the model
    misc_id: string, optional
             identifier of the 'misc file' that contains meta-information
    mask_id: string, optional
             file id of the mask image       
    paradigm_id: string, optional
                 identifier of the paradigm file (should be a .csv file)
    contrast_id: string, optional
                 id of the contrast file
    design_id: string, optional
               id of the design matrices file   
    glm_dump: string, optional,
              id of the glm dump file (should be .npz file)
    glm_config: string, optional,
                id of the glm config file (should disappear in the near future)
    
    Returns
    -------
    paths, dictionary
        containing all the paths that are required to perform a
        glm with brainvisa
    """
 
    paths = {}
    paths['minf'] = os.sep.join(( base_path, "Minf"))
    paths['model'] = os.sep.join(( base_path, "glm", model_id))
    if paradigm_id is not None:
        paths['paradigm'] = os.sep.join(( paths['minf'], paradigm_id))
        if not os.path.isfile( paths['paradigm']):
            raise ValueError,"paradigm file %s not found" % paths['paradigm']

    paths['mask'] = os.sep.join(( paths['minf'], mask_id))
    paths['misc'] = os.sep.join(( paths['minf'], misc_id))
    paths['contrast_file'] =  os.sep.join(( paths['model'], contrast_id))
    paths['contrasts'] = os.sep.join(( paths['model'], "Contrast"))
    if not os.path.exists(paths["contrasts"]):
        os.makedirs( paths["contrasts"])
        
    paths['fmri'] = {}
    paths['dmtx'] = {}
    paths['glm_dump'] = {}
    paths['glm_config'] = {}
    for sess in sessions:
        designPath = os.sep.join(( paths['model'], sess))
        if not os.path.exists(designPath):
            os.makedirs(designPath)
        paths['dmtx'][sess] = os.sep.join(( designPath, design_id))  
        fmriPath = os.sep.join(( base_path, sess))
        paths['fmri'][sess] = glob.glob( os.sep.join((fmriPath, fmri_wc)))
        if  paths['fmri'][sess]==[]:
            print "found no fMRI file as %s" %os.sep.join((fmriPath, fmri_wc))
        else:
            paths['fmri'][sess].sort()
        paths['glm_dump'][sess] = os.sep.join((designPath, glm_dump))
        paths['glm_config'][sess] = os.sep.join((designPath, glm_config))
    return paths

def generate_brainvisa_ouput_paths(
    output_dir_path, contrasts, z_file=True, stat_file=True, con_file=True,
    res_file=True, html_file=True):
    """
    This function generate standard output paths for all the contrasts
    and arranges them in a dictionary

    Parameters
    ----------
    output_dir_path: string,
                     path of the output dir
    contrasts: ConfigObj instance,
              contrast_structure
    z_file: bool, optional
            whether the z_file should be written or not
    stat_file: bool, optional
               whether the stat file (t or F) should be written or not
    con_file: bool, optional,
              whether the contrast file should be written or not
    res_file: bool, optional
              whether the residual variance file should be written or not
    html_file: bool, optional,
              whether the html result page should be written or not  
    """
    paths={}
    contrast_ids = contrasts["contrast"]
    for c in contrast_ids:
        paths[c]={}
        if z_file:
            paths[c]["z_file"] = os.sep.join(( output_dir_path, "%s_%s.nii"%\
                                               (str(c), "z_map")))
        if stat_file:
            # there is a switch fere between t/F files
            contrast_type = contrasts[c]["Type"]
            if contrast_type == "t":
                paths[c]["stat_file"] = os.sep.join(( output_dir_path,
                                                      "%s_T_map.nii"% str(c)))
            elif contrast_type == "F":
                paths[c]["stat_file"] = os.sep.join(( output_dir_path,
                                                      "%s_F_map.nii"% str(c)))        
        if res_file:
            paths[c]["res_file"] = os.sep.join(( output_dir_path,
                                                 "%s_ResMS.nii"% str(c)))
        if con_file:
            paths[c]["con_file"] = os.sep.join(( output_dir_path,
                                                 "%s_con.nii"%str(c)))
        if html_file:
            paths[c]["html_file"] = os.sep.join(( output_dir_path, "%s.html"%\
                                                  str(c)))
    return paths

################################################
# IO functions
################################################


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
           if mask is not None:
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

def save_volume_from_mask_image(path, mask_image, data, descrip=None):
    """
    volume saving utility for masked volumes
    
    Parameters
    ----------
    path, string, output image path
    mask_image, string,
               path of ther reference mask image
    data=None data to be put in the volume
    descrip=None, a string descibing what the image is
    """
    mask = load(mask_image)
    shape = mask.get_shape()
    affine = mask.get_affine()
    mdata = mask.get_data()
    save_volume(shape, path, affine, mdata, data, descrip)
    
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
       print "Could not write the image: no mask"
       return

    if data == None:
       print "Could not write the image: no data"
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
    Parameters
    ----------
    contrast a structure describing 
             the values related to the computed contrast 
    ContrastId, string, the contrast identifier
    dim the dimension of the contrast
    mask_url path of the mask image related to the data
    kargs, might have 'z_file', 'stat_file', 'con_file', 'res_file', 'html_file'
           keys yielding paths to write corresponding outputs.
           optionally it can also have the keys 
           'method', 'threshold' and 'cluster'
           that are used to define the parameters for vizualization 
           of the html page. 
    """
    z_file = None
    stat_file = None
    res_file = None
    con_file = None
    html_file = None
    if kargs.has_key("z_file"):
        z_file = kargs["z_file"]
    if kargs.has_key("stat_file"):
        stat_file = kargs["stat_file"]
    if kargs.has_key("res_file"):
        res_file = kargs["res_file"]
    if kargs.has_key("con_file"):
        con_file = kargs["con_file"]
    if kargs.has_key("html_file"):
        html_file = kargs["html_file"]
    mask = load(mask_url)
    mask_arr = mask.get_data()
    affine = mask.get_affine()
    shape = mask.get_shape()    
   
    # load the values
    t = contrast.stat()
    z = contrast.zscore()

    # saving the Z statistics map
    if z_file is not None:
        save_volume(shape, z_file, affine, mask_arr, z, "z_file")
    
    # Saving the t/F statistics map
    if stat_file is not None:
        save_volume(shape, stat_file, affine, mask_arr, t, "stat_file")
    
    if int(dim) != 1:
        shape = (shape[0], shape[1], shape[2], int(dim)**2)
        contrast.variance = contrast.variance.reshape(int(dim)**2, -1)

    ## saving the associated variance map
    # fixme : breaks with F contrasts !
    if z_file is not None:
        if contrast.type == "t":
            save_volume(shape, res_file, affine, mask_arr,
                        contrast.variance)
        if int(dim) != 1:
            shape = (shape[0], shape[1], shape[2], int(dim))

    # writing the associated contrast estimate
    # fixme : breaks with F contrasts !
    if con_file is not None:
        if contrast.type == "t":
            save_volume(shape, con_file, affine, mask_arr, contrast.effect)
            
    # writing the results as an html page
    if html_file is not None:
        import html_result
        method = 'fpr'
        threshold = 0.001
        cluster = 0
        if kargs.has_key("method"):
            method = kargs["method"]
   
        if kargs.has_key("threshold"):
            threshold = kargs["threshold"]

        if kargs.has_key("cluster"):
            cluster = kargs["cluster"]
    
        html_result.display_results_html(z_file, mask_url, html_file,
                                         threshold=threshold,
                                         method=method, cluster_th=cluster)

######################################################
# First Level analysis
######################################################

#-----------------------------------------------------
#------- Design Matrix handling ----------------------
#-----------------------------------------------------

def design_matrix(
    misc_file, output_file, session,  paradigm_file, frametimes,
    hrf_model="Canonical", drift_model="Blank", add_regs=None, drift_order=2,
    hfcut=128, fir_delays=[0], fir_duration=1., model="default",
    add_reg_names=None, verbose=0):
    """
    Estimation of the design matrix and update of misc info

    Parameters
    ----------
    misc_file: string,
              path of misc info file that is updated with info on design matrix
    output_file: string,
                 path of the (.csv) file
                 where the design matrix shall be written
    session: string,
             id of the session        
    paradigm_file: string, 
                   path of (.csv) paradigm-describing file
                   or None, if no such file exists
    concerning the following parameters, please refer to 
    nipy.neurospin.utils.design_matrix
    
    Returns
    -------
    dmtx: nipy.neurospin.utils.design_matrix.DesignMatrix
          instance
    """
    import nipy.neurospin.utils.design_matrix as dm

    # get the condition names
    misc = ConfigObj(misc_file)
    if session.isdigit():
        _session = int(session)
    else:
        _session = misc["sessions"].index(session)

    # get the paradigm
    if isinstance(paradigm_file, basestring):
        _paradigm = dm.load_protocol_from_csv_file(paradigm_file, _session)
    else:
        _paradigm = None

    # compute the design matrix
    dmtx = dm.DesignMatrix(frametimes, _paradigm, hrf_model=hrf_model,
         drift_model=drift_model, hfcut=hfcut, drift_order=drift_order,
         fir_delays=fir_delays, fir_duration=fir_duration,
         add_regs=add_regs, add_reg_names=add_reg_names)
    dmtx.estimate()

    # write the design matrix
    dmtx.write_csv(output_file)

    # write some info in the misc file
    if not misc.has_key(model):
        misc[model] = {}
    misc[model]["regressors_%s" % session] = dmtx.names
    misc[model]["design matrix cond"] = dmtx.design_cond
    misc.write()
    return dmtx


#-----------------------------------------------------
#------- GLM fit -------------------------------------
#-----------------------------------------------------

def glm_fit(fMRI_path, DesignMatrix,  output_glm=None, glm_info=None,
           fit="Kalman_AR1", mask_url=None, data_scaling=True):
    """
    Call the GLM Fit function with apropriate arguments

    Parameters
    ----------
    fMRI_path: string or list of strings,
               path of the fMRI data file(s)
    design_matrix: DesignMatrix instance,
                   design matrix of the model
    output_glm: string, optional
                path of the output glm .npz dump
    glm_info: string,optional
               path of the output configobj  that gives dome infor on the glm
    fit: string, Optional,
         to be chosen among 'Kalman_AR1', 'Ordinary Least Squares', 'Kalman'
         that represents both the model and the fit method
    mask_url: string, Optional,
              path of the mask file
             if None, no mask is applied
    data_scaling: bool, Optional
                  scaling of the data to mean value

    Returns
    -------
    glm, a nipy.neurospin.glm.glm instance representing the GLM

    Note
    ----
    either DesignMatrix or design_matrix_path have to be defined
    
    fixme: mask should be optional
    """
    import nipy.neurospin.glm

    # get the model/fit methods
    if fit == "Kalman_AR1":
        model = "ar1"
        method = "kalman"
    elif fit == "Ordinary Least Squares":
        method = "ols"
        model="spherical"
    elif fit == "Kalman":
        method = "kalman"
        model = "spherical"

    # get the design matrix
    if isinstance(DesignMatrix, basestring):
        import nipy.neurospin.utils.design_matrix as dm
        X = dm.dmtx_from_csv( DesignMatrix).matrix
    else:
        X = DesignMatrix.matrix

    # load the fMRI data
    Y = load_image(fMRI_path, mask_url)

    # data_scaling to percent of mean, and mean removal
    if data_scaling:
        # divide each voxel time course by its mean value, subtract 1,
        # mulitply by 100 to deal with percent of average BOLD fluctuations 
        mY = np.repeat(np.expand_dims(Y.mean(-1), -1), Y.shape[-1], Y.ndim-1)
        Y = 100* (Y/mY - 1)

    # apply the GLM
    glm = nipy.neurospin.glm.glm()
    glm.fit(Y.T, X, method=method, model=model)

    # Write outputs
    if output_glm is not None:
        glm.save(output_glm)
        
    if glm_info is not None:
        cobj = ConfigObj(glm_info)
        cobj["DesignMatrix"] = X
        cobj["mask_url"] = mask_url
        cobj["GlmDumpFile"] = output_glm
        cobj.write()   

    return glm


#-----------------------------------------------------
#------- Contrast computation ------------------------
#-----------------------------------------------------


def compute_contrasts(contrast_struct, misc, CompletePaths, glms=None,
                     model="default", **kargs):
    """
    Contrast computation utility    
    
    Parameters
    ----------
    contrast_struct, ConfigObj instance or string
               it yields the set of contrasts of the multi-session model
               or the path to a configobj that specifies the contrasts
    misc: misc object instance,
              misc information on the datasets used here
              or path to a configobj file that yields the misc info
    Complete_Paths: dictionary or string,
                    yields all paths, indexed by contrasts,
                    where outputs will be written
                    if it is a string, all the paths are re-generated 
                    based on it as a output directory
    glms, dictionary of nipy.neurospin.glm.glm.glm instances
         indexed by sessions, optional
         if it is not provided, a 'glm_config' instance should be provided
         in kargs
    model='default', string,
                     name of the contrast model used in miscfile
    """
    
    # read the misc info
    if isinstance(misc, basestring):
        misc = ConfigObj(misc)
    if not misc.has_key(model):
        misc[model] = {}
    if not misc[model].has_key("con_dofs"):
        misc[model]["con_dofs"] = {}
    sessions = misc['sessions']
    
    # get the contrasts
    if isinstance(contrast_struct, basestring):
        contrast_struct = ConfigObj(contrast_struct) 
    contrasts_names = contrast_struct["contrast"]

    # get the glms
    designs = {}
    if glms is not None:
       designs = glms
    else:             
        if not kargs.has_key('glm_config'):
            raise ValueError, "No glms provided"
        else:
            import nipy.neurospin.glm
            for s in sessions:
                try:
                    designs[s] = nipy.neurospin.glm.load(
                        kargs['glm_config'][s]["GlmDumpFile"])
                except:
                    print "glm could not be loaded for session %s, \
                           expect errors" %s

    # set the mask
    mask_url = None

    if misc.has_key('mask_url'): mask_url = misc['mask_url']
    if contrast_struct.has_key('mask_url'):
        mask_url = contrast_struct['mask_url']
        
    # set the output paths
    if isinstance(CompletePaths, basestring) :
        CompletePaths = generate_brainvisa_ouput_paths(CompletePaths, 
                        contrast_struct)
    # compute the contrasts
    for i, contrast in enumerate(contrasts_names):
        try:
            contrast_type = contrast_struct[contrast]["Type"]
            contrast_dimension = contrast_struct[contrast]["Dimension"]
            final_contrast = []
            multicon = dict()

            for key, value in contrast_struct[contrast].items():
                if key not in ["Type", "Dimension"]:
                    session = "_".join(key.split("_")[:-1])
                    bv = np.asarray([int(j) for j in value])
                    if contrast_type == "t" and bv.any():
                        _con = designs[session].contrast(bv.astype(np.float))
                        final_contrast.append(_con)

                    if contrast_type == "F":
                        if not multicon.has_key(session):
                            multicon[session] = bv.astype(np.float)
                        else:
                            multicon[session] = np.vstack((
                                multicon[session], bv.astype(np.float)))
            if contrast_type == "F":
                for key, value in multicon.items():
                    if sum([j != 0 for j in value.reshape(-1)]) != 0:
                        _con = designs[key].contrast(value)    
                        final_contrast.append(_con)
        
            res_contrast = final_contrast[0]
            for c in final_contrast[1:]:
                res_contrast = res_contrast + c
                res_contrast.type = contrast_type
            
            # write misc information
            cpp = CompletePaths[contrast]
            if kargs.has_key('cluster'): cpp['cluster'] = kargs['cluster']
            if kargs.has_key('threshold'): cpp['threshold'] = kargs['threshold']
            if kargs.has_key('method'): cpp['method'] = kargs['method']
            save_all_images(res_contrast, contrast_dimension, mask_url, cpp)
            misc[model]["con_dofs"][contrast] = res_contrast.dof
        #except ValueError,
        #    'contrast %s does not fit into memory -- skipped' %contrast
        except:
            import sys
            print "Unexpected error:", sys.exc_info()[0]
            raise
    misc.write()


