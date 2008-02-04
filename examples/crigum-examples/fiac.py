from neuroimaging.core.image.image import Image

prefix = "http://kff.stanford.edu"

anat = Image("%s/FIAC/avganat.img" % prefix)
avg152 = Image("%s/FIAC/avg152T1_brain.img" % prefix)
fmri = Image("%s/FIAC/fiac3/fonc3/fsl/filtered_func_data.img" % prefix)


# subject 3's sentence and overall 

prefix = "http://www-stat.stanford.edu/~jtaylo/matthew"
sentence = Image("%s/fixed-spm/block/contrasts/sentence/fiac3/t.nii" % prefix)
average = Image("%s/fixed-spm/block/contrasts/average/fiac3/t.nii" % prefix)



# random effects sentence and overall 

prefix = "http://kff.stanford.edu/FIAC/multi/block/contrasts"
sentence = Image("%s/sentence/t.nii" % prefix)
average = Image("%s/average/t.nii" % prefix)

