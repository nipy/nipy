import nifti
import numpy as np
subject = 's12920'
for i in range(5,133):
    nim = nifti.NiftiImage('/volatile/thirion/Localizer/%s/fMRI/acquisition/loc1/swaloc1_corr%04d.img' %(subject,i))
    nim2 = nifti.NiftiImage(nim.data,nim.header)
    nim2.save('/volatile/thirion/Localizer/%s/fMRI/acquisition/loc1/swaloc1_corr%04d.nii' %(subject,i))
    
