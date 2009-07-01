"""
Example of a demo that fits a GMM to  a dataset


Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
import numpy.random as nr
import nipy.neurospin.clustering.gmm as gmm

# this should be put elsewhere
def plot2D(x,my_gmm,z = None,show = 0,verbose=0):
    """
    Given a set of points in a plane and a GMM, plot them
    INPUT:
    - x: array of shape (npoints,dim=2)
    - my_gmm: a gmm whose density has to be ploted
    - z=None: array of shape (npoints)
    that gives a labelling of the points in x
    by default, it is not taken into account
    - show = 0: do we show the image
    - verbose = 0 : verbosity mode
    
    my_gmm should have a method 'likelihood' that
    takes an array of points of shape (np,dim)
    and returns an array of shape (np,my_gmm.k)
    that represents  the likelihood component-wise 
    """
    if x.shape[1]!= my_gmm.dim:
        raise ValueError, 'Incompatible dimension between data and model'
    if x.shape[1]!=2:
        raise ValueError, 'this works only for 2D cases'
    
    gd1 = gmm.grid_descriptor(2)
    xmin = x.min(0); xmax = x.max(0)
    xm = 1.1*xmin[0]-0.1*xmax[0]
    xs = 1.1*xmax[0]-0.1*xmin[0]
    ym = 1.1*xmin[1]-0.1*xmax[1]
    ys = 1.1*xmax[1]-0.1*xmin[1]
    
    gd1.getinfo([xm,xs,ym,ys],[51,51])
    grid = gd1.make_grid()
    L = my_gmm.likelihood(grid).sum(1)   
    if verbose: print L.sum()*(xs-xm)*(ys-ym)/2500

    import matplotlib.pylab as mp
    mp.figure()
    gdx = gd1.nbs[0]
    Pdens= np.reshape(L,(gdx,np.size(L)/gdx))
    mp.imshow(np.transpose(Pdens),alpha = 2.0,
              origin ='lower',extent=[xm,xs,ym,ys])
 
    if z==None:
        mp.plot(x[:,0],x[:,1],'o')
    else:
        import matplotlib as ml
        hsv = ml.cm.hsv(range(256)) 
        col = hsv[range(0,256,256/int(z.max()+1)),:]
        for k in range(z.max()+1):
            mp.plot(x[z==k,0],x[z==k,1],'o',color=col[k])   
           
    mp.axis([xm,xs,ym,ys])
    mp.colorbar()
    if show: mp.show()


dim = 2
# 1. generate a 3-components mixture
x1 = nr.randn(100,dim)
x2 = 3+2*nr.randn(50,dim)
x3 = np.repeat(np.array([-2,2],ndmin=2),30,0)+0.5*nr.randn(30,dim)
x = np.concatenate((x1,x2,x3))

#2. fit the mixture with a bunch of possible models
krange = range(1,10)
lgmm = gmm.best_fitting_GMM(x,krange,prec_type='diag',niter=100,delta = 1.e-4,ninit=1,verbose=0)

# 3, plot the result
z = lgmm.map_label(x)
plot2D(x,lgmm,z,show = 1,verbose=0)
