# Example usage
# python compute_mask.py swa4D.nii.gz mask.nii.gz
# python compute_mask.py swa*.nii mask.nii.gz

from fff2.utils.mask import compute_mask_files
import sys

if len(sys.argv) < 3:
	print """Usage : python compute_mask [ -s copyfilename ] inputfilename(s) outputfilename
inputfilename can be either a single (4D) file or a list of (3D) files
-s copyfilename : also save a copy of the orginal data as a single 4D file named copyfilename

# Example :
python compute_mask.py swa4D.nii.gz mask.nii.gz
python compute_mask.py swa*.nii mask.nii.gz
python compute_mask.py -s swaCopy4D.nii.gz swa*.nii ../masks/mask.nii.gz"""
	sys.exit()
copyIn4Dfilename = None
if '-s' in sys.argv:
	i = sys.argv.index('-s')
	sys.argv.pop(i)
	copyIn4Dfilename = sys.argv.pop(i)
if len(sys.argv) == 3:
	inputFilename = sys.argv[1]
else:
	inputFilename = sorted(sys.argv[1:-1])
outputFilename = sys.argv[-1]
compute_mask_files(inputFilename, outputFilename, copy_filename = copyIn4Dfilename)
