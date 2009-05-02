from _transform_affine import * 


def preconditioner(radius):
    """
    Computes a scaling vector pc such that, if p=(u,r,s,q) represents
    affine transformation parameters, where u is a translation, r and
    q are rotation vectors, and s is a scaling vector, then all
    components of (p/pc) are somehow comparable and homogeneous to the
    distance unit implied by the translation component.
    """
    rad = 1./radius
    sca = radius
    return np.array([1,1,1,rad,rad,rad,sca,sca,sca,rad,rad,rad])


def vector12_to_param(t, precond, stamp): 
    """
    param = vector12_to_param(t, precond, stamp)
    """
    param = t/precond

    if stamp == transformation_types['rigid 3D']:
        param = param[0:6]
    elif stamp == transformation_types['similarity 3D']:
        param = param[0:7] 
    elif stamp == transformation_types['rigid 2D']:
        param = param[[0,1,5]]
    elif stamp == transformation_types['similarity 2D']:
        param = param[[0,1,5,6,7]]
    elif stamp == transformation_types['affine 2D']:
        param = param[[0,1,5,6,7,11]]

    return param


def transform(xyz, T):
    """
    XYZ = transform(xyz, T)

    T is a 4x4 matrix.
    xyz is a 3xN array of 3d coordinates stored row-wise.  
    """
    XYZ = np.dot(T[0:3,0:3], xyz)
    XYZ[0,:] += T[0,3]
    XYZ[1,:] += T[1,3]
    XYZ[2,:] += T[2,3]
    return XYZ 
