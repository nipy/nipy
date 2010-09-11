# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
eneral tools to analyse fMRI datasets (GLM fit)
sampled on the surface.

This relies massively on glm_tools, and on the tio module
to perform io on (AIMS .tex) textures.
This should be replaced in a near future by pygifti  modules.

Author : Bertrand Thirion, 2010
"""
import os
import glob
import numpy as np

from ...externals.configobj import ConfigObj

from . import tio


############################################
# Path definition
############################################

def generate_all_brainvisa_paths( base_path, sessions, fmri_wc,  model_id,
                                  misc_id="misc_info.con", 
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
        containing all the paths that are required
        to perform a glm with brainvisa
    """
 
    paths = {}
    paths['minf'] = os.sep.join(( base_path, "Minf"))
    paths['model'] = os.sep.join(( base_path, "glm", model_id))
    if paradigm_id is not None:
        paths['paradigm'] = os.sep.join(( paths['minf'], paradigm_id))
        if not os.path.isfile( paths['paradigm']):
            raise ValueError,"paradigm file %s not found" %paths['paradigm']
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

def generate_brainvisa_ouput_paths( output_dir_path, contrasts, side,
                                    z_file=True, stat_file=True, con_file=True,
                                    res_file=True):
    """
    This function generate standard output paths for all the contrasts
    and arranges them in a dictionary

    Parameters
    ----------
    output_dir_path: string,
                     path of the output dir
    contrasts: ConfigObj instance,
              contrast_structure
    side: string,
          'left' or 'right' (as these are yusually in separate files)
          this will be a prefix to all paths
    z_file: bool, optional
            whether the z_file should be written or not
    stat_file: bool, optional
               whether the stat file (t or F) should be written or not
    con_file: bool, optional,
              whether the contrast file should be written or not
    res_file: bool, optional
              whether the residual variance file should be written or not

    Returns
    -------
    path, a dictiorany with all paths
    """
    paths={}
    contrast_ids = contrasts["contrast"]
    for c in contrast_ids:
        paths[c]={}
        if z_file:
            paths[c]["z_file"] = os.sep.join(
                ( output_dir_path, "%s_%s_z_map.tex"% (side, str(c))))
        if stat_file:
            # there is a switch fere between t/F files
            contrast_type = contrasts[c]["Type"]
            if contrast_type == "t":
                paths[c]["stat_file"] = os.sep.join(
                    (output_dir_path, "%s_%s_T_map.tex"% (side, str(c))))
            elif contrast_type == "F":
                paths[c]["stat_file"] = os.sep.join(
                    (output_dir_path, "%s_%s_F_map.tex"% (side, str(c))))
        if res_file:
            paths[c]["res_file"] = os.sep.join(
                (output_dir_path, "%s_%s_ResMS.tex"% (side, str(c))))
        if con_file:
            paths[c]["con_file"] = os.sep.join(
                ( output_dir_path, "%s_%s_con.tex" % (side, str(c))))
 
    return paths

################################################
# IO functions
################################################


def load_texture(path):
    """
    Return an array from texture data

    Parameters
    ----------
    path string or list of strings
         path of the texture files

    Returns
    -------
    data array of shape (nnode) or (nnode, len(path)) 
         the orresponding data
    """
    if hasattr(path, '__iter__'):
       if len(path)==1:
          path = path[0]

    if hasattr(path, '__iter__'):
        Fun = []
        for f in path:
            Fun.append(tio.Texture(f).read(f).data)
        Fun = np.array(Fun)
    else:
        Fun = tio.Texture(path).read(path).data

    return Fun


def save_texture(path, data):
    """
    volume saving utility for textures
    
    Parameters
    ----------
    path, string, output image path
    data, array of shape (nnode)
          data to be put in the volume

    Fixme
    -----
    Missing checks
    Handle the case where data is multi-dimensional ? 
    """
    tio.Texture(path, data=data).write(path)

################################################
# GLM and contrasts
################################################


def glm_fit(fMRI_path, DesignMatrix=None,  output_glm=None, outputCon=None,
           fit="Kalman_AR1", design_matrix_path=None):
    """
    Call the GLM Fit function with apropriate arguments

    Parameters
    ----------
    fMRI_path, string or list of strings,
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
    design_marix_path: string,
                       path of the design matrix .csv file

    Returns
    -------
    glm, a nipy.neurospin.glm.glm instance representing the GLM

    Note
    ----
    either DesignMatrix or design_matrix_path have to be defined
    """
    import nipy.neurospin.glm
    
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
        #X = dm.DesignMatrix().read_from_csv(DesignMatrix).matrix
    else:
        X = DesignMatrix.matrix
  
    Y = load_texture(fMRI_path)

    glm = nipy.neurospin.glm.glm()
    glm.fit(Y, X, method=method, model=model)

    if output_glm is not None:
        glm.save(output_glm)
        
    if outputCon is not None:
        cobj = ConfigObj(outputCon)
        cobj["DesignMatrix"] = X
        cobj["GlmDumpFile"] = output_glm
        cobj.write()   

    return glm

def compute_contrasts(contrast_struct, misc, CompletePaths, glms=None,
                     model="default", **kargs):
    """
    Contrast computation utility    
    
    Parameters
    ----------
    contrast_struct, ConfigObj instance or string
               it yields the set of contrasts of the multi-session model
               or the path to a configobj that specifies the contarsts
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
    """
    
    # read the msic info
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
            import nipy.neurospin.glm as GLM
            for s in sessions:
                designs[s] = GLM.load(kargs['glm_config'][s]["GlmDumpFile"])

    # set the output paths
    if isinstance(CompletePaths, basestring) :
        CompletePaths = generate_brainvisa_ouput_paths(CompletePaths, 
                        contrast_struct)
    # compute the contrasts
    for i, contrast in enumerate(contrasts_names):
        contrast_type = contrast_struct[contrast]["Type"]
        contrast_dimension = contrast_struct[contrast]["Dimension"]
        final_contrast = []
        multicon = dict()

        for key, value in contrast_struct[contrast].items():
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

        res_contrast = final_contrast[0]
        for c in final_contrast[1:]:
            res_contrast = res_contrast + c
            res_contrast.type = contrast_type
            
        # write misc information
        cpp = CompletePaths[contrast]
        save_all_textures(res_contrast, contrast_dimension, cpp)
        misc[model]["con_dofs"][contrast] = res_contrast.dof
    
    misc.write()

def save_all_textures(contrast, dim,  kargs):
    """
    idem savel_all, but the names are now all included in kargs
    """
    z_file = kargs["z_file"]
    t_file = kargs["stat_file"]
    res_file = kargs["res_file"]
    con_file = kargs["con_file"]
     
    # load the values
    t = contrast.stat()
    z = contrast.zscore()

    # saving the Z statistics map
    save_texture(z_file, z)
    
    # Saving the t/F statistics map
    save_texture(t_file, t)
    
    ## saving the associated variance map
    # fixme : breaks with F contrasts !
    if contrast.type == "t":
        save_texture(res_file, contrast.variance)

    # writing the associated contrast structure
    # fixme : breaks with F contrasts !
    if contrast.type == "t":
        save_texture(con_file, contrast.effect)
