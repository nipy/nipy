import scipy.ndimage as sn
import numpy as np
import commands
import os

# ---------------------------------------------
# various FSL-based Pre processings functions -
# ---------------------------------------------

def SliceTiming(file, tr, outputFile, interleaved = False,
                ascending = True):
    """
    Perform slice timing using FSL
    """
    so = " "
    inter = " "
    if interleaved:
        inter = "--odd"
    if not ascending:
        so = "--down"
    print "slicetimer -i '%s' -o '%s' %s %s -r %s" % \
          (file, outputFile, so, inter, str(tr))
    print commands.getoutput("slicetimer -i '%s' -o '%s' %s %s -r %s" %\
                             (file, outputFile, so, inter, str(tr)))

def Realign(file, refFile, outputFile):
    """
    Perform realignment using FSL
    """
    print commands.getoutput("mcflirt -in '%s' -out '%s' -reffile '%s'\
    -mats" % (file, outputFile, refFile))

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
    print commands.getoutput("flirt -in '%s' -ref '%s' -omat '%s' \
    -out '%s' -bins 1024 -cost corratio %s -dof 12" \
                             % (anat, templatet1, norm_matrix, normAnat, s1) )
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
    fmri = load(file)
    voxel_width = fmri.get_header['pixdim'][2]
    sigma = fwhm/(voxel_width*2*sqrt(2*log(2)))
    for i in fmri.data:
        sn.gaussian_filter(i, sigma, order=0, output=None,
                           mode='reflect', cval=0.0)
    fmri.save(outputFile)
