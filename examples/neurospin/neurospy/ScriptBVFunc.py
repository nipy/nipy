from numpy import *
import commands
import nifti
## For Smoothing
import scipy.ndimage as SN
## For GLM
from vba import VBA
## For Contrast Computation
from configobj import ConfigObj
import Results
## For Mask Computation
from fff2.utils.mask import compute_mask_intra 
## For the tools
import os

#########
# Tools #
#########

def save_volume(volume, file, header, mask=None, data=None):
  if mask != None and data != None:
    if size(data.shape) == 1:
      volume[mask > 0] = data
    else:
      for i in range(data.shape[0]):
        volume[i][mask[0] > 0] = data[i]
    nifti.NiftiImage(volume,header).save(file)

def saveall(contrast, design, ContrastId, dim, kargs):
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
    mask_arr= mask_arr.reshape(1, mask_arr.shape[0], mask_arr.shape[1], mask_arr.shape[2])
  shape = mask_arr.shape
  t = contrast.stat()
  z = contrast.zscore()
  results = "Z map"
  z_file = os.sep.join((contrasts_path, "%s_%s.nii"% (str(ContrastId), paths[results])))
  save_volume(zeros(shape), z_file, header, mask_arr, z)
  if contrast.type == "t":
    results = "Student-t tests"
  elif contrast.type == "F":
    results = "Fisher tests"
  t_file = os.sep.join((contrasts_path, "%s_%s.nii" % (str(ContrastId), paths[results])))
  save_volume(zeros(shape), t_file, header, mask_arr, t)
  if int(dim) != 1:
    shape = (int(dim) * int(dim), shape[1], shape[2], shape[3])
    contrast.variance = contrast.variance.reshape(int(dim) * int(dim), -1)
  results = "Residual variance"
  res_file = os.sep.join((contrasts_path, "%s_%s.nii" % (str(ContrastId), paths[results])))
  save_volume(zeros(shape), res_file, header, mask_arr, contrast.variance)
  if int(dim) != 1:
    shape = (int(dim), shape[1], shape[2], shape[3])
  results = "contrast definition"
  con_file = os.sep.join((contrasts_path, "%s_%s.nii" % (str(ContrastId), paths[results])))
  save_volume(zeros(shape), con_file, header, mask_arr, contrast.effect)
  if kargs.has_key("method"):
    method = kargs["method"]
  else:
    print "Cannot save HTML results. Missing argument : method"
    return
  if kargs.has_key("threshold"):
    threshold = kargs["threshold"]
  else:
    print "Cannot save HTML results. Missing argument : threshold"
    return
  if kargs.has_key("cluster"):
    cluster = kargs["cluster"]
  else:
    cluster = 0
  results = "HTML Results"
  html_file = os.sep.join((contrasts_path, "%s_%s.html" % (str(ContrastId), paths[results])))
  Results.ComputeResultsContents(z_file, design.mask_url, html_file, threshold = threshold, method = method, cluster = cluster)


def ComputeMask(fmriFiles, outputFile, infT = 0.2, supT = 0.9):
    compute_mask_intra(fmriFiles, outputFile, False,None, infT, supT)

###################
# Pre processings #
###################

def SliceTiming(file, tr, outputFile, interleaved = False, ascending = True):
  so = " "
  inter = " "
  if interleaved:
    inter = "--odd"
  if not ascending:
    so = "--down"
  print "slicetimer -i '%s' -o '%s' %s %s -r %s" % (file, outputFile, so, inter, str(tr))
  print commands.getoutput("slicetimer -i '%s' -o '%s' %s %s -r %s" % (file, outputFile, so, inter, str(tr)))

def Realign(file, refFile, outputFile):
  print commands.getoutput("mcflirt -in '%s' -out '%s' -reffile '%s' -mats" % (file, outputFile, refFile))

def NormalizeAnat(anat, templatet1, normAnat, norm_matrix, searcht1 = "NASO"):
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
  #  voxel_width = 3
  fmri = nifti.NiftiImage(file)
  #voxel_width = fmri.header['voxel_size'][2]
  voxel_width = fmri.header['pixdim'][2]
  sigma = fwhm/(voxel_width*2*sqrt(2*log(2)))
  for i in fmri.data:
    SN.gaussian_filter(i, sigma, order=0, output=None, mode='reflect', cval=0.0)
  fmri.save(outputFile)


########################
# First Level analysis #
########################

def DesignMatrix(nbFrames, paradigm, miscFile, tr, outputFile, session, hrf = "Canonical", drift = "Blank", driftMatrix = None, poly_order = 2, cos_FreqCut = 128, FIR_order = 1, FIR_length = 1, model = "default"):
  ## For DesignMatrix
  import DesignMatrix as DM
  from dataFrame import DF

  design = DM.DesignMatrix(nbFrames, paradigm, session, miscFile, model)
  design.load()
  design.timing(tr)
  if driftMatrix != None:
    drift = pylab.load(driftMatrix)
  elif drift == "Blank":
    drift = 0
  elif drift == "Cosine":
    DesignMatrix.HF = cos_FreqCut
    drift = DM.cosine_drift
  elif drift == "Polynomial":
    DesignMatrix.order = poly_order
    drift = DM.canonical_drift
  if hrf == "Canonical":
    hrf = DM.hrf.glover
  elif hrf == "Canonical With Derivative":
    hrf = DM.hrf.glover_deriv
  elif hrf == "FIR Model":
    design.compute_fir_design(drift = drift, name = session, o = FIR_order, l = FIR_length)
    output = DF(colnames=design.names, data=design._design)
    output.write(outputFile)
    return 0
  else:
    print "Not HRF model passed. Aborting process."
    return
  design.compute_design(hrf = hrf, drift = drift, name = session)
  if hasattr(design, "names"):
    output = DF(colnames=design.names, data=design._design)
    print design.names
    output.write(outputFile)

def GLMFit(file, designMatrix, mask, outputVBA, outputCon, fit = "Kalman_AR1"):
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
  s["HDF5FilePath"] = outputVBA
  s["ConfigFilePath"] = outputCon
  s["DesignFilePath"] = designMatrix
  glm.save(s)
  return glm


#def ComputeContrasts(contrastFile, miscFile, designs, paths, save_mode="Contrast Name"):
def ComputeContrasts(contrastFile, miscFile, glms, savefunc, save_mode="Contrast Name", model = "default", **kargs):
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
    savefunc(res_contrast, design, ContrastId, contrast_dimension, kargs)
    misc[model]["con_dofs"][contrast] = res_contrast.dof
  misc["Contrast Save Mode"] = save_mode
  misc.write()

##################
# Group Analysis #
##################
