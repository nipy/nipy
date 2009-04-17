import numpy as np

import volumeimages as v
import nipy.neurospin.statistical_mapping as sm

def remake_images(): 
    # Get group data
    f = np.load('data/offset_002.npz')
    data, vardata, xyz = f['mat'], f['var'], f['xyz']
    
    dX = xyz[0,:].max() + 1
    dY = xyz[1,:].max() + 1
    dZ = xyz[2,:].max() + 1
    
    aux = np.zeros([dX,dY,dZ])
    
    data_images = []
    vardata_images = []
    mask_images = []
    for i in range(data.shape[0]):
        aux[list(xyz)] = data[i,:]
        data_images.append(v.nifti1.Nifti1Image(aux.copy(), np.eye(4)))
        aux[list(xyz)] = vardata[i,:]
        vardata_images.append(v.nifti1.Nifti1Image(aux.copy(), np.eye(4)))
        aux[list(xyz)] = 1
        mask_images.append(v.nifti1.Nifti1Image(aux.copy(), np.eye(4)))

    return data_images, vardata_images, mask_images

data_images, vardata_images, mask_images = remake_images()
##zimg, mask = sm.onesample_test(data_images, None, mask_images, 'student')

zimg, mask, nulls = sm.onesample_test(data_images, None, mask_images, 'wilcoxon', 
                                      permutations=1024, cluster_forming_th=0.01)
clusters, info = sm.cluster_stats(zimg, mask, 0.01, nulls)
