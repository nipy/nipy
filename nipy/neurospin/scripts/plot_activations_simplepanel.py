""" Short yet flexible script to quickly plot 2D slices of an image, optionally adding another image as overlay
    Uses pynifti. Requires also scipy if the overlay need to be resampled. """
from matplotlib import pyplot
from numpy import dot, isnan, any, allclose, diag, r_, vstack, hstack, ma
import nifti

panelise = lambda allimgs, cnum = 4: vstack([ hstack(allimgs[slice(i*cnum,(i+1)*cnum)]) for i in range(len(allimgs)/cnum) ])
volume_to_panel = lambda img, num = 16, cnum = 4 : panelise([img[x] for x in r_[.2:.8:num*1j] * len(img)], cnum)

# default viewnum values: 0:axial, 1:sagittal, 2:coronal. Left display Left.
def orient_voxelview(img, viewnum, viewdef = [('IS', 'AP', 'LR'), ('LR', 'SI', 'PA'), ('PA', 'SI', 'LR')]):
	imgax = [x[0] for x in img.getSOrientation(True)[::-1]]
	if 'U' in imgax: return img.data.squeeze().T # unknown orientation, can't do much
	a, f = zip(*[[(i, (1,-1)[v[0]!=x]) for i, x in enumerate(imgax) if x in v][0] for v in viewdef[viewnum]])
	return img.data[...,None,None,None].swapaxes(a[0], 3).swapaxes(a[1], 4).swapaxes(a[2], 5)[0,0,0,::f[0],::f[1],::f[2]]

def slicer(figure, t1, atlas, viewnum, roismin = [0], roismax = [1], title = None, vmin=None, vmax=None, alpha=0.75, num=16, cnum=4):
	ax = figure.add_subplot(111)
	imgPanelised = volume_to_panel(orient_voxelview(t1, viewnum), num, cnum)
	figure.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0,  wspace = 0, hspace = 0)
	i = ax.imshow(imgPanelised, cmap=pyplot.cm.gray, origin='upper', aspect='auto', vmin=vmin, vmax=vmax)
	if atlas:
		atlaspanel = volume_to_panel(orient_voxelview(atlas, viewnum), num, cnum)
		maskedAtlas = ma.masked_array(atlaspanel, mask = (atlaspanel < roismin) | (atlaspanel > roismax))
		ax.imshow(maskedAtlas, cmap=pyplot.cm.autumn, interpolation='nearest', origin='upper', aspect='auto', alpha=alpha)
	if title:
		t=ax.text(3, 3, title, color='white', va='top', ha='left', fontsize=12)

def resample_NiftiImage(src, target, order = 3, mode = 'constant'):
	from scipy.ndimage import affine_transform
	affine = dot(src.sform_inv, target.sform)
	af, of = affine[:3, :3], affine[:3, -1]
	srcdata = src.data if not any(isnan(src.data)) else nan_to_num(src.data)
	if allclose(af, diag(diag(af))): # if scaling matrix
		af = af.diagonal()
		of /= af
	a = affine_transform(srcdata.T, af, of, output_shape=target.data.T.shape, order = order, mode = mode).T
	return nifti.NiftiImage(a, target.header)

if __name__ == "__main__":
	import sys
	idx = [i for i, x in enumerate(sys.argv) if x.startswith('-')] + [len(sys.argv)]
	args = dict([(sys.argv[idx[i]], sys.argv[idx[i]+1:idx[i+1]]) for i in range(len(idx)-1)])
        if (not '-o' in args) or (not '-i' in args) or (len (args["-i"])==0):
                sys.exit( """Mandatory arguments:
  -i filename.nii : input image
  -o filename.png : output png/ps file
Optionals arguments : (figw/figh : screen size ; num/colnum : screen layout; view : view Axis number ; vmin/vmax : visual range ; etc.)
Example using all optionals arguments :
  python %s -i Localizer_meananat.nii  -o out.png  -overlay Localizer_con31_smoothed_thresh_2_detected_T.nii -overlaymin 0.1 -overlaymax 100 -figw 8 -figh 8 -num 16 -colnum 4 -view 0 -dpi 64 -alpha 0.75 -title filename -vmin 0 -vmax 1500 #-noresample""" % __file__)
	infile, outfile = args["-i"], args["-o"]
	viewnum = int(args.get("-view", [0])[0])
	num, cnum = int(args.get("-num", [16])[0]), int(args.get("-colnum", [4])[0])
	overlay = args.get("-overlay", None)
	overlaymin, overlaymax = float(args.get("-overlaymin", [0])[0]), float(args.get("-overlaymax", [0])[0])
	noresample = "-noresample" in args
	dpi = int(args.get("-dpi", [64])[0])
	figh, figw = int(args.get("-figh", [8])[0]), int(args.get("-figw", [8])[0])
	vmin = float(args["-vmin"][0]) if "-vmin" in args else None
	vmax = float(args["-vmax"][0]) if "-vmax" in args else None
	title = args.get("-title", [None])[0]
	t1 = nifti.NiftiImage(infile[0])
	atlas = overlay and nifti.NiftiImage(overlay[0])
	if overlay and atlas and not noresample:
		atlas = resample_NiftiImage(atlas, t1, order = 0)
	f = pyplot.figure(figsize=(figw, figh), dpi=dpi)
	title = atlas.filename.rsplit('/')[-1] if (title == "filename" and atlas) else title
	slicer(f, t1, atlas, viewnum, roismin = overlaymin, roismax = overlaymax, title = title, vmin=vmin, vmax=vmax, alpha = float(args.get("-alpha", [0.75])[0]), num=num, cnum=cnum)
	f.savefig(outfile[0], dpi=dpi)