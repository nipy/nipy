from neuroimaging.core.image.image import Image

prefix = "http://kff.stanford.edu"

anat = Image("%s/FIAC/avganat.img" % prefix)
avg152 = Image("%s/FIAC/avg152T1_brain.img" % prefix)
fmri = Image("%s/FIAC/fiac3/fonc3/fsl/filtered_func_data.img" % prefix)
#sentence = Image("sentence.nii")
#average = Image("average.nii")