"""This file contains functions to display a large picture of a bunch of brain slices, using pylab and pynifti.
It can be used as a module, or a standalone program. In the later case, see the usage information at the __main__ below.
TODO : add an option for non-contour mode and for grid+markers mode"""

from numpy import *
import nifti, sys
from pylab import *
import os

def resampleNiftiImage(src, target, order = 2, mode = 'constant'):
	from scipy.ndimage import affine_transform
	affine = dot(src.sform_inv, target.sform)
	af, of = affine[:3, :3], affine[:3, -1]
	a = affine_transform(src.data.T, af, of, output_shape=target.data.T.shape, order = order, mode = mode).T
	return nifti.NiftiImage(a, target.header)


def showOneView(niImage, view, snum, contourNiTemplate = None, nameAsLabels = True, labelcolor = 'white', contourcolor = 'yellow', numContour = 3, **kargs):
	"""This matplotlib functions display the slice 'snum' of image 'niImage', for pov 'view' (0 or 'axial',
1 or 'sagittal', 2 or 'coronal'), with a contour overlay (useful to check Registrations).
'contourNiTemplate', if not None, is an image from which the contour overlay is computed, and is
assumed to be the same shape as 'niImage', for example, an MNI brainmask. 
'nameAsLabels' decorates the axis with english words (such as 'left' or 'anterior') instead
of numerical values"""
	ijk = swapaxes(niImage.data, 0, 2).astype(float)
	axesnames = niImage.getSOrientation(True)
	templateijk = swapaxes((contourNiTemplate.data if contourNiTemplate else niImage.data), 0, 2) # avoid specialcasing
	if view == "axial" or view == 0 or view == '0':
		axPrim = [("Inferior" in x) for x in axesnames].index(True)
		axesnames.pop(axPrim)
		axSec = [("Left" in x) for x in axesnames].index(True)
		v = rollaxis(rollaxis(ijk, axPrim)[snum], axSec, 2)
		tv = rollaxis(rollaxis(templateijk, axPrim)[snum], axSec, 2)
		nameJ = axesnames.pop(axSec)
		nameI = axesnames.pop(0)
	if view == "sagittal" or view == 1 or view == '1':
		axPrim = [("Left" in x) for x in axesnames].index(True)
		axesnames.pop(axPrim)
		axSec = [("Inferior" in x) for x in axesnames].index(True)
		v = rollaxis(rollaxis(ijk, axPrim)[snum], axSec, 0)
		tv = rollaxis(rollaxis(templateijk, axPrim)[snum], axSec, 0)
		nameI = axesnames.pop(axSec)
		nameJ = axesnames.pop(0)
	if view == "coronal" or view == 2 or view == '2':
		axPrim = [("Anterior" in x) for x in axesnames].index(True)
		axesnames.pop(axPrim)
		axSec = [("Left" in x) for x in axesnames].index(True)
		v = rollaxis(rollaxis(ijk, axPrim)[snum], axSec, 2)
		tv = rollaxis(rollaxis(templateijk, axPrim)[snum], axSec, 2)
		nameJ = axesnames.pop(axSec)
		nameI = axesnames.pop(0)
	# cosmetics :
	ax = gca().matshow(v.copy(), **kargs).axes
	if contourNiTemplate:
		ax.contour(tv.copy(), numContour, colors=contourcolor, alpha = .80)
	xticks(color=labelcolor, fontsize='smaller')
	yticks(color=labelcolor, fontsize='smaller', rotation = 90)
	if nameAsLabels:
		l = [''] * len(ax.get_xticks())
		l[1], _, l[-2] = nameJ.split('-')
		ax.set_xticklabels(l)
		l = [''] * len(ax.get_yticks())
		l[1], _, l[-2] = nameI.split('-')
		ax.set_yticklabels(l)
	else:
		xlabel(nameJ, color=labelcolor)
		ylabel(nameI, color=labelcolor)
	# switch visual orientation so that Left is always displayed on Left, and Inferior at bottom
	if nameI == 'Inferior-to-Superior' or nameI == 'Posterior-to-Anterior':
		if ax.get_ylim()[1] <= 0.:
			ax.set_ylim(ax.get_ylim()[::-1])
	if nameJ == 'Right-to-Left' or nameJ == 'Anterior-to-Posterior':
		if ax.get_xlim()[1] > 0.:
			ax.set_xlim(ax.get_xlim()[::-1])



def display_norm_contour_manyfilenames(niTemplate, filenameStruct, filenameBold, views = [0, 1, 2], angles = [66, 48, 46]):
	"""Display many brains slices on a large matplotlib figure.
	
	niTemplate : a nifti image defining the "template" size on which other images will be
	visually resampled, and the contour of which will be overlayed. An in-Talairach mask is ideal here.
	filenameStruct : a list of structural (or anything else) images filepaths, the view of which
	is to be drawn on the left part of the final image. They should lie in the same CS as the template.
	filenameBold : a list of BOLD (or anything else) images filepaths, to be drawn on the right, or
	None if not needed. They should also lie on the same coordinate system as the template
	views : a list of views needed (with : 0 or "axial", 1 or "sagittal", 2 or "coronal")
	angles : a list of slices (respectively on each above-mentionned axes). Caveat : as those are
	related to the template image, be sure to update it when switching to a different templates.
	
	Note : the implementation is slower than necessary as the same contour is computed many times"""
	N = len(filenameStruct)
	assert (filenameBold == None or len(filenameBold) == N)
	colnum = (2*len(views)) if filenameBold != None else len(views)
	# some cosmetics
	f = figure(figsize=(4 * len(views) * (2 if filenameBold != None else 1), 4 * N), dpi=72, facecolor='k')
	f.subplotpars.bottom, f.subplotpars.top = 0.01, 0.99
	f.subplotpars.left, f.subplotpars.right = 0.02, 1-0.01
	f.subplotpars.hspace, f.subplotpars.wspace = 0.4, 0.01
	matplotlib.interactive(False)
	for j in range(N):
		print "drawing %s (%d/%d)" % (filenameStruct[j], j+1, N)
		try:
			niRStruct = resampleNiftiImage(nifti.NiftiImage(filenameStruct[j]), niTemplate)
		except:
			print "Error opening/resampling %s"  % filenameStruct[j]
			niRStruct = nifti.NiftiImage(zeros_like(niTemplate.data),niTemplate.header)
		for i in range(len(views)):
			ax = subplot(N, colnum, j*colnum + i + 1 )
			showOneView(niRStruct, views[i], angles[i], niTemplate, cmap=cm.gray)
			ax.text(8, -10, "(%d)" % angles[i], color='yellow', fontsize='smaller')
			if i == 1:
				ax.set_title(("%d " % j) + filenameStruct[j] + " \n", color='w')
		if filenameBold == None:
			continue
		try:
			niRBold = resampleNiftiImage(nifti.NiftiImage(filenameBold[j]), niTemplate)
		except:
			print "Error opening/resampling %s"  % filenameBold[j]
			niRBold = nifti.NiftiImage(zeros_like(niTemplate.data), niTemplate.header)
		for i in range(len(views)):
			ax = subplot(N, colnum, j*colnum + len(views) + i + 1 )
			showOneView(niRBold, views[i], angles[i], niTemplate, cmap=cm.gray)
	matplotlib.interactive(True)


import glob
if __name__ == '__main__':
	idx = [i for i, x in enumerate(sys.argv) if x.startswith('-')] + [len(sys.argv)]
	args = dict([(sys.argv[idx[i]], sys.argv[idx[i]+1:idx[i+1]]) for i in range(len(idx)-1)])
	if (not '-o' in args) or (not '-i' in args) or (len (args["-i"])==0):
		print """Usage:
	-i : input image list
	-o : output png/ps file
	-s : optional second image list, to be displayed on a second column set.
	-mask : mask image (if not, use first image from the -i list)
	-views and -angles : defines which slices to display. Must be two lists
	 of same size, the first defining the view (0 or axial, 1 or sagittal,
	 2 or coronal), the second defining the slice number (in the mask voxel
	 dimension)

Examples :
slices_ploter.py -mask MNI152_T1_1mm_brain_mask.nii.gz -i anat_subject1.nii anat_subject2.nii -s bold_subject1.nii bold_subject2.nii -views 0 1 2 -angles 66 48 46 -o output_twocolumns.png
slices_ploter.py -mask MNI152_2mm_mask.nii -i anat_subject1.nii anat_subject2.nii -views axial -angles 33 -o /tmp/out_singlecolumn_axial33only.png
slices_ploter.py -i struct.nii.gz -s struct2.nii.gz -o two_cols.png"""
		sys.exit()
	outpath = args["-o"][0]
	filenamesS = args["-i"]
	filenamesB = args["-s"] if "-s" in args else None
	assert(filenamesB == None or len(filenamesS) == len(filenamesB))
	try:
		views, angles = args["-views"], [int(x) for x in args["-angles"]]
	except KeyError:
		views, angles = [0, 1, 2], [33, 24, 23]
	maskImg = args['-mask'][0] if "-mask" in args else filenamesS[0]
	k = nifti.NiftiImage(maskImg)
	display_norm_contour_manyfilenames(k, filenamesS, filenamesB, views = views, angles = angles)
	
	print ("saving %s" % outpath)
	savefig(outpath, dpi=58, facecolor='k', edgecolor='k', transparent=False)