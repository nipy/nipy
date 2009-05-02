
import nipy.neurospin as fff2
import nipy.neurospin
import nipy.neurospin.glm as glm

import numpy as np 


def affect_inmask(dest, src, xyz):
    if xyz == None:
        dest = src
    else:
        dest[xyz[0,:], xyz[1,:], xyz[2,:]] = src
    return dest



class linear_model: 

    def __init__(self, data=None, design_matrix=None, mask=None, formula=None, 
                 model='spherical', method=None, niter=2):

        if data == None:
            self.data = None
            self.xyz = None
            self.glm = None

        else:
            if not isinstance(data, fff2.neuro.image):
                raise ValueError, 'Invalid input image.'
            if not isinstance(design_matrix, np.ndarray):
                raise ValueError, 'Invalid design matrix.'
            
            self.data = data
            if mask == None:
                self.xyz = None
                Y = data.array
                axis = 3
            else:
                if not isinstance(mask, fff2.neuro.image):
                    raise ValueError, 'Invalid mask image.'
                self.xyz = np.where(mask.array>0)
                Y = data.array[self.xyz]
                axis = 1
                
            self.glm = glm.glm(Y, design_matrix, formula=formula, axis=axis, model=model, method=method, niter=niter)


    def dump(self, filename):
        """
        Dump GLM fit as NPZ file.  
        """
        self.glm.save(filename)


    def contrast(self, vector):
        """
        Compute images of contrast and contrast variance.  
        """
        c = self.glm.contrast(vector)
        
        con = np.zeros(self.data.array.shape[1:4])
        con_img = fff2.neuro.image(affect_inmask(con, c.effect, self.xyz), transform=self.data.transform)

        vcon = np.zeros(self.data.array.shape[1:4])
        vcon_img = fff2.neuro.image(affect_inmask(vcon, c.variance, self.xyz), transform=self.data.transform)

        dof = c.dof
        
        return con_img, vcon_img, dof


