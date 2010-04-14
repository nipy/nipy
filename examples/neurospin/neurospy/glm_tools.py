"""
General tools to analyse fMRI datasets (GLM fit)
using nipy.neurospin tools

Author : Lise Favre, Bertrand Thirion, 2008-2010
"""

import numpy as np
import commands
import os
from configobj import ConfigObj

from nipy.io.imageformats import load, save, Nifti1Image 
import glob
from os.path import join



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
        containing all the paths that are required to eprform a glm with brainvisa
    """
 
    paths = {}
    paths['minf'] = os.sep.join(( base_path, "Minf"))
    paths['model'] = os.sep.join(( base_path, "glm", model_id))
    paths['paradigm'] = os.sep.join(( paths['minf'], paradigm_id))
    if not os.path.isfile( paths['paradigm']):
            raise ValueError,"paradigm file %s not found" %paradigmFile
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

def generate_brainvisa_ouput_paths( output_dir_path, contrasts, z_file=True,
                                    stat_file=True, con_file=True, res_file=True,
                                    html_file=True):
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
                paths[c]["stat_file"] = os.sep.join(( output_dir_path, "%s_%s.nii"%\
                                                   (str(c), "T_map")))
            elif contrast_type == "F":
                paths[c]["stat_file"] = os.sep.join(( output_dir_path, "%s_%s.nii"%\
                                                   (str(c), "F_map")))        
        if res_file:
            paths[c]["res_file"] = os.sep.join(( output_dir_path, "%s_%s.nii"%\
                                                 (str(c), "ResMS")))
        if con_file:
            paths[c]["con_file"] = os.sep.join(( output_dir_path, "%s_%s.nii"%\
                                                 (str(c), "con")))
        if html_file:
            paths[c]["html_file"] = os.sep.join(( output_dir_path, "%s.html"%\
                                                  (str(c))))
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
    save_volume(shape, z_file, affine, mask_arr, z, "z_file")
    
    # Saving the t/F statistics map
    save_volume(shape, stat_file, affine, mask_arr, t, "stat_file")
    
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

    import html_result
    html_result.ComputeResultsContents(z_file, mask_url, html_file,
                                   threshold=threshold, method=method,
                                   cluster=cluster)

######################################################
# First Level analysis
######################################################

#-----------------------------------------------------
#------- Design Matrix handling ----------------------
#-----------------------------------------------------

def _loadProtocol(path, session):
    """
    Read a paradigm file consisting of a list of pairs
    (occurence time, (duration), event ID)
    and create a paradigm array
    
    Parameters
    ----------
    path, string a path to a .csv file that describes the paradigm
    session, int, the session number used to extract 
             the relevant session information in th csv file
    
    Returns
    -------
    paradigm array of shape (nevents,2) if the type is event-related design 
             or (nenvets,3) for a block design
             that constains (condition id, onset) 
             or (condition id, onset, duration)
    """
    import csv
    csvfile = open(path)
    dialect = csv.Sniffer().sniff(csvfile.read())
    csvfile.seek(0)
    reader = csv.reader(open(path, "rb"),dialect)
    
    paradigm = []
    for row in reader:
        paradigm.append([float(row[j]) for j in range(len(row))])

    paradigm = np.array(paradigm)
    
    #paradigm = loadtxt(path)
    if paradigm[paradigm[:,0] == session].tolist() == []:
        return None
    paradigm = paradigm[paradigm[:,0] == session]
    
    if paradigm.shape[1] == 4:
        paradigm = paradigm[:,1:]
        typep = 'block'
    else:
        typep ='event'
        paradigm = paradigm[:,[1,2]]
    
    return paradigm

def DesignMatrix(nbFrames, paradigm_file, miscFile, tr, outputFile,
                  session, hrfType="Canonical", drift="Blank",
                  regMatrix=None, poly_order=2, cos_FreqCut=128,
                  FIR_delays=[0], FIR_duration=1., model="default",
                  regNames=None, verbose=0):
    """
    """
    import nipy.neurospin.utils.design_matrix as dm

    # set the frametimes
    _frametimes = tr*np.arange(nbFrames)

    # get the condition names
    misc = ConfigObj(miscFile)
    if session.isdigit():
        _session = int(session)
    else:
        _session = misc["sessions"].index(session)
    _names  = misc["tasks"]

    # get the paradigm
    _paradigm = _loadProtocol(paradigm_file, _session)

    # compute the design matrix
    DM = dm.DesignMatrix\
        (_frametimes, _paradigm, hrf_model=hrfType,
         drift_model=drift, hfcut=cos_FreqCut, drift_order=poly_order,
         fir_delays=FIR_delays, fir_duration=FIR_duration, cond_ids=_names,
         add_regs=regMatrix, add_reg_names=regNames)
    DM.estimate()

    # write the design matrix
    DM.write_csv(outputFile)

    # write some info in the misc file
    if not misc.has_key(model):
        misc[model] = {}
    misc[model]["regressors_%s" % session] = DM.names
    misc[model]["design matrix cond"] = DM.design_cond
    misc.write()
    return DM


#-----------------------------------------------------
#------- GLM fit -------------------------------------
#-----------------------------------------------------

def GLMFit(file, DesignMatrix=None,  output_glm=None, outputCon=None,
           fit="Kalman_AR1", mask_url=None, design_matrix_path=None):
    """
    Call the GLM Fit function with apropriate arguments

    Parameters
    ----------
    file, string or list of strings,
          path of the fMRI data file(s)
    design_matrix, DesignMatrix instance, optional
          design matrix of the model
    output_glm, string, optional
                path of the output glm .npz dump
    outputCon, string,optional
               path of the output configobj contrast object
    fit= 'Kalman_AR1', string to be chosen among
         "Kalman_AR1", "Ordinary Least Squares", "Kalman"
         that represents both the model and the fit method
    mask_url=None string, path of the mask file
             if None, no mask is applied
    design_marix_path: string,
                       path of the design matrix .csv file

    Returns
    -------
    glm, a nipy.neurospin.glm.glm instance representing the GLM

    Note
    ----
    either DesignMatrix or design_matrix_path have to be defined
    
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

    # get the design matrix
    if isinstance(DesignMatrix, basestring):
        import nipy.neurospin.utils.design_matrix as dm
        X = dm.DesignMatrix().read_from_csv(DesignMatrix).matrix
    else:
        X = DesignMatrix.matrix
  
    Y = load_image(file, mask_url)

    import nipy.neurospin.glm as GLM
    glm = GLM.glm()
    glm.fit(Y.T, X, method=method, model=model)

    if output_glm is not None:
        glm.save(output_glm)
        
    if outputCon is not None:
        cobj = ConfigObj(outputCon)
        cobj["DesignMatrix"] = X
        cobj["mask_url"] = mask_url
        cobj["GlmDumpFile"] = output_glm
        cobj.write()   

    return glm


#-----------------------------------------------------
#------- Contrast computation ------------------------
#-----------------------------------------------------


def ComputeContrasts(contrasts=None, misc=None, glms=None,
                     save_mode="Contrast Name", model="default",
                     verbose=0, **kargs):
    """
    Contrast computation utility    
    
    Parameters
    ----------
    contrasts, ConfigObj instance, optional
               it yields the set of contrasts of the multi-session model
               if None, a 'contrast_file' (path) should be provided
    miscFile, misc object instance, optional,
              misc information on the datasets used here
              if None, a 'misc_file' (path) should be provided
    glms, dictionary of nipy.neurospin.glm.glm.glm instances
         indexed by sessions, optional
         if it is not provided, a 'glm_config' should be provided in kargs
    save_mode='Contrast Name', string to be chosen
                        among 'Contrast Name' or 'Contrast Number'
    model='default', string,
                     name of the contrast model used in miscfile
    """
    import nipy.neurospin.glm as GLM

    #read the msic info
    if misc==None:
        if not kargs.has_key('misc_file'):
            misc = ConfigObj(kargs['miscFile'])
        misc = ConfigObj(miscFile)
    if not misc.has_key(model):
        misc[model] = {}
    if not misc[model].has_key("con_dofs"):
        misc[model]["con_dofs"] = {}

    # get the contrasts
    if contrasts==None:
        if not kargs.has_key('contrast_file'):
            raise ValueError, "No contrast file provided"
        contrasts = ConfigObj(kargs['contrast_file'])
    contrasts_names = contrasts["contrast"]

    # get the glms
    designs = {}
    sessions = misc['sessions']
    if glms is not None:
       designs = glms
    else:             
        if not kargs.has_key('glms_config'):
            raise ValueError, "No glms provided"
        else:
            for s in sessions:
                designs[s] = GLM.load(kargs['glm_config'][s]["GlmDumpFile"])
                
    if misc.has_key('mask_url'):
        mask_url = misc['mask_url']
    else:
        mask_url = None

    if not kargs.has_key('CompletePaths'):
        raise ValueError, 'write paths are not available'

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
                bv = [int(j) != 0 for j in value]
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
            
        cpp = kargs['CompletePaths'][contrast]
        save_all_images(res_contrast, contrast_dimension, mask_url, cpp)
                                          
        misc[model]["con_dofs"][contrast] = res_contrast.dof
    misc["Contrast Save Mode"] = save_mode
    misc.write()


