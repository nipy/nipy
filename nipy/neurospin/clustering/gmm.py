"""
Gaussian Mixture Model Class:
contains the basic fields and methods of GMMs
The class GMM _old uses C bindings which are
computationally and memory efficient.

Author : Bertrand Thirion, 2006-2009
"""

import numpy as np
import nipy.neurospin.clustering.clustering as fc
from numpy.random import rand

class grid_descriptor():
    """
    A tiny class to handle cartesian grids
    """
    def __init__(self,dim=1):
        self.dim = dim

    def getinfo(self,lim,nbs):
        if len(lim)==2*self.dim:
            self.lim = lim
        else: raise ValueError, "Wrong dimension for grid definition"
        if np.size(nbs)==self.dim:
            self.nbs = nbs
        else: raise ValueError, "Wrong dimension for grid definition"

    def make_grid(self):
        size = np.prod(self.nbs)
        grid = np.zeros((size,self.dim))
        grange = []

        for j in range(self.dim):
            xm = self.lim[2*j]
            xM = self.lim[2*j+1]
            if np.isscalar(self.nbs):
                xb = self.nbs
            else:
                xb = self.nbs[j]
            zb = size/xb
            gr = xm +float(xM-xm)/(xb-1)*np.arange(xb).astype('f')
            grange.append(gr)

        if self.dim==1:
            grid = np.array([[grange[0][i]] for i in range(xb)])

        if self.dim==2:
            for i in range(self.nbs[0]):
                for j in range(self.nbs[1]):
                    grid[i*self.nbs[1]+j,:]= np.array([grange[0][i],grange[1][j]])

        if self.dim==3:
            for i in range(self.nbs[0]):
                for j in range(self.nbs[1]):
                    for k in range(self.nbs[2]):
                        q = (i*self.nbs[1]+j)*self.nbs[2]+k
                        grid[q,:]= np.array([grange[0][i],grange[1][j],grange[2][k]])
        if self.dim>3:
            print "Not implemented yet"
        return grid

def best_fitting_GMM(x,krange,prec_type='full',niter=100,delta = 1.e-4,ninit=1,verbose=0):
    """
    Given a certain dataset x, find the best-fitting GMM
    within a certain range indexed by krange

    INPUT:
    -----
    - x array of shape (nbitem,dim)
        the data from which the model is estimated
    - krange (list of floats) the range of values to test for k
    - prec_type='full' the vocariance parameterization
        (to be chosen within 'full','diag') for full
        and diagonal covariance respectively
    - niter=100: maximal number of iterations in the estimation process
    - delta = 1.e-4: increment of data likelihood at which
        convergence is declared
    - ninit=1: number of initialization performed
        to reach a good solution
    - verbose=0: verbosity mode
    
    OUPUT:
    -----
    - mg : the best-fitting GMM
    """
    if np.size(x) == x.shape[0]:
        x = np.reshape(x,(np.size(x),1))

    dim = x.shape[1]
    bestbic = -np.infty
    for k in krange:
        lgmm = GMM(k,dim,prec_type)
        gmmk = lgmm.initialize_and_estimate(x,None,niter,delta,ninit,verbose)
        bic = gmmk.evidence(x)
        if bic>bestbic:
            bestbic = bic
            bgmm = gmmk
        if verbose:
            print 'k', k,'bic',bic 
    return bgmm


class GMM():
    """
    Standard GMM.

    this class contains the following fields
    - k (int): the number of components in the mixture
    - dim (int): is the dimension of the data
    - prec_type='full' (string) is the parameterization
    of the precisions/covariance matrices:
    either 'full' or 'diagonal'.
    - means array of shape (k,dim):
    all the means (mean parameters) of the components
    - precisions array of shape (k,dim,dim):
    the precisions (inverse covariance matrix) of the components    
    - weights: array of shape(k): weights of the mixture

    fixme :
    - naming conventions ? 
    - no copy method
    - leave open the possibility to  call the c library
    """

    def __init__(self, k=1, dim=1, prec_type='full', means = None,
                 precisions=None, weights=None):
        """
        Initialize the structure, at least with the dimensions of the problem
        At most, with what is necessary to compute the likelihood of a point
        under the model
        """
        self.k = k
        self.dim = dim
        self.prec_type=prec_type
        self.means = means
        self.precisions = precisions
        self.weights = weights

        if self.means==None:
            self.means = np.zeros((self.k,self.dim))

        if self.precisions==None:
            if prec_type=='full':
                prec = np.reshape(np.eye(self.dim),(1,self.dim,self.dim))
                self.precisions = np.repeat(prec,self.k,0)
            else:
                self.precisions = np.ones((self.k,self.dim))
            
        if self.weights==None:
            self.weights = np.ones(self.k)*1.0/self.k

    def plugin(self,means=None,precisions=None,weights = None):
        """
        sets manually the main fields of the bgmm
        """
        self.means = means
        self.precisions = precisions
        self.weights = weights
        self.check()
    
    def check(self):
        """
        Checking the shape of sifferent matrices involved in the model
        """
        if self.means.shape[0] != self.k:
            raise ValueError," self.means does not have correct dimensions"
            
        if self.means.shape[1] != self.dim:
            raise ValueError," self.means does not have correct dimensions"

        if self.weights.size != self.k:
            raise ValueError," self.weights does not have correct dimensions"
        
        if self.dim !=  self.precisions.shape[1]:
            raise ValueError, "self.precisions does not have correct dimensions"

        if self.prec_type=='full':
            if self.dim !=  self.precisions.shape[2]:
                raise ValueError, "self.precisions does not have correct dimensions"

        if self.prec_type=='diag':
            if np.shape(self.precisions) !=  np.shape(self.means):
                raise ValueError, "self.precisions does not have correct dimensions"

        if self.precisions.shape[0] != self.k:
            raise ValueError,"self.precisions does not have correct dimensions"

        if self.prec_type not in ['full','diag']:
            raise ValueError, 'unknown precisions type'

    def check_x(self,x):
        """
        essentially check that x.shape[1]==self.dim
        """
        if np.size(x)==x.shape[0]:
            x = np.reshape(x,(np.size(x),1))
        if x.shape[1]!=self.dim:
            raise ValueError, 'incorrect size for x'
        return x

    def init(self,x):
        """
        this function initializes self according to a certain dataset x:
        1. sets the hyper-parameters
        2. initializes z using a k-means algorithm, then
        3. upate the parameters
        INPUT:
        - x: array of shape (nbitems,self.dim)
        the data used in the estimation process
        """
        import nipy.neurospin.clustering.clustering as fc
        n = x.shape[0]
        
        #1. set the priors
        self.guess_regularizing(x,bcheck=1)

        # 2. initialize the memberships
        if self.k>1:
            cent,z,J = fc.cmeans(x,self.k)
        else:
            z = np.zeros(n).astype(np.int)
        
        l = np.zeros((n,self.k))
        l[np.arange(n),z]=1

        #3.update the parameters
        self.update(x,l)
    
    def pop(self,l,tiny = 1.e-15):
        """
        compute the population, i.e. the statistics of allocation
        INPUT:
        - l array of shape (nbitem,self.k):
        the likelihood of each item being in each class
        """
        sl = np.maximum(tiny,np.sum(l,1))
        nl = (l.T/sl).T
        return np.sum(nl,0)
        
    def update(self,x,l):
        """
        Identical to self._Mstep(x,l)
        """
        self._Mstep(x,l)
        

    def likelihood(self,x):
        """
        return the likelihood of the model for the data x
        the valeus are weighted by the components weights
        INPUT:
        - x:array of shape (nbitems,self.dim)
        the data used in the estimation process
         OUPUT:
        - l array of shape(nbitem,self.k)
        component-wise likelihood
        """
        l = self.unweighted_likelihood(x)
        l *= self.weights
        return l

    def unweighted_likelihood(self,x):
        """
        return the likelihood of each data for each component
        Using Normal model
        the values are not weighted by the component weights
         INPUT:
        - x: array of shape (nbitems,self.dim)
        the data used in the estimation process
         OUPUT:
        - l array of shape(nbitem,self.k)
        unwieghted component-wise likelihood
        """
        n = x.shape[0]
        l = np.zeros((n,self.k))
        from numpy.linalg import det

        for k in range(self.k):
            # compute the data-independent factor first
            w = - np.log(2*np.pi)*self.dim
            m = np.reshape(self.means[k],(1,self.dim))
            b = self.precisions[k]
            if self.prec_type=='full':
                w += np.log(det(b))
                q = np.sum(np.dot(m-x,b)*(m-x),1)
            else:
                w += np.sum(np.log(b))
                q = np.dot((m-x)**2,b)
            w -= q
            w /= 2
            l[:,k] = np.exp(w)   
        return l
    
    def mixture_likelihood(self,x):
        """
        returns the likelihood of the mixture for x
         INPUT:
        - x: array of shape (nbitems,self.dim)
        the data used in the estimation process
        """
        x = self.check_x(x)
        l = self.likelihood(x)
        sl = np.sum(l,1)
        return sl

    def average_log_like(self,x,tiny = 1.e-15):
        """
        returns the likelihodd of the mixture for x
        INPUT:
        tiny=1.e-15: a small constant to avoid numerical singularities
        """
        x = self.check_x(x)
        l = self.likelihood(x)
        sl = np.sum(l,1)
        sl = np.maximum(sl,tiny)
        return np.mean(np.log(sl))

    def evidence(self,x):
        """
        computation of bic approximation of evidence
        INPUT:
        - x array of shape (nbitems,dim)
        the data from which bic is computed
        OUPUT:
        - the bic value
        """
        x = self.check_x(x)
        tiny = 1.e-15
        l = self.likelihood(x)
        return self.bic(l,tiny)
    
    def bic(self,l = None,tiny = 1.e-15):
        """
        computation of bic approximation of evidence
        INPUT:
        - l: array of shape (nbitem,self.k)
        component-wise likelihood
        if l==None,  it is re-computed in E-step
        - tiny=1.e-15: a small constant to avoid numerical singularities
        OUPUT:
        the bic value
        """
        sl = np.sum(l,1)
        sl = np.maximum(sl,tiny)
        bicc  = np.sum(np.log(sl))
        
        # number of parameters
        n = l.shape[0]
        if self.prec_type=='full':
            eta = self.k*(1 + self.dim + (self.dim*self.dim+1)/2)-1
        else:
            eta = self.k*(1 + 2*self.dim )-1
        bicc = bicc-np.log(n)*eta
        return bicc

    def _Estep(self,x):
        """
        E step
        returns the likelihood per class of each data item
        INPUT:
        - x array of shape (nbitems,dim)
        the data used in the estimation process
        OUPUT:
        - l array of shape(nbitem,self.k)
        component-wise likelihood
        """
        return self.likelihood(x)

    def guess_regularizing(self,x,bcheck=1):
        """
        Set the regularizing priors as weakly informative
        according to Fraley and raftery;
        Journal of Classification 24:155-181 (2007)
        """
        small = 0.01
        # the mean of the data
        mx = np.reshape(x.mean(0),(1,self.dim))

        dx = x-mx
        vx = np.dot(dx.T,dx)/x.shape[0]
        if self.prec_type=='full':
            px = np.reshape(np.diag(1.0/np.diag(vx)),(1,self.dim,self.dim))
        else:
            px =  np.reshape(1.0/np.diag(vx),(1,self.dim))
        px *= np.exp(2.0/self.dim*np.log(self.k))
        self.prior_means = np.repeat(mx,self.k,0)
        self.prior_weights = np.ones(self.k)/self.k
        self.prior_scale = np.repeat(px,self.k,0)
        self.prior_dof = self.dim+2
        self.prior_shrinkage = small
        self.weights = np.ones(self.k)*1.0/self.k
        if bcheck:
            self.check()
    
    def _Mstep(self,x,l):
        """
        M step regularized according to the procedure of
        Fraley et al. 2007
        - x: array of shape(nbitem,self.dim)
        the data from which the model is estimated
        - l: array of shape(nbitem,self.k)
        the likelihood of the data under each class
        """
        from numpy.linalg import pinv
        tiny  =1.e-15
        pop = self.pop(l)
        sl = np.maximum(tiny,np.sum(l,1))
        l = (l.T/sl).T
        
        # shrinkage,weights,dof
        self.weights = self.prior_weights + pop
        self.weights = self.weights/(self.weights.sum())
        
        #reshape
        pop = np.reshape(pop,(self.k,1))
        prior_shrinkage = self.prior_shrinkage
        shrinkage = pop + prior_shrinkage

        # means
        means = np.dot(l.T,x)+ self.prior_means*prior_shrinkage
        self.means= means/shrinkage
        
        #precisions
        empmeans = np.dot(l.T,x)/np.maximum(pop,tiny)
        empcov = np.zeros(np.shape(self.precisions))
        
        if self.prec_type=='full':
            for k in range(self.k):
                dx = x-empmeans[k]
                empcov[k] = np.dot(dx.T,l[:,k:k+1]*dx) 
                    
            covariance = np.array([pinv(self.prior_scale[k])
                                   for k in range(self.k)])
            covariance += empcov

            dx = np.reshape(empmeans-self.prior_means,(self.k,self.dim,1))
            addcov = np.array([np.dot(dx[k],dx[k].T) for k in range(self.k)])
        
            apms =  np.reshape(prior_shrinkage*pop/shrinkage,(self.k,1,1))
            covariance += addcov*apms

            dof = self.prior_dof+pop+self.dim+2
            covariance /= np.reshape(dof,(self.k,1,1))
        
            self.precisions = np.array([pinv(covariance[k]) \
                                       for k in range(self.k)])
        else:
            for k in range(self.k):
                dx = x-empmeans[k]
                empcov[k] = np.sum(dx**2*l[:,k:k+1],0) 
                    
            covariance = np.array([1.0/(self.prior_scale[k])
                                   for k in range(self.k)])
            covariance += empcov

            dx = np.reshape(empmeans-self.prior_means,(self.k,self.dim,1))
            addcov = np.array([np.sum(dx[k]**2,0) for k in range(self.k)])

            apms =  np.reshape(prior_shrinkage*pop/shrinkage,(self.k,1))
            covariance += addcov*apms

            dof = self.prior_dof+pop+self.dim+2
            covariance /= np.reshape(dof,(self.k,1))
        
            self.precisions = np.array([1.0/covariance[k] \
                                       for k in range(self.k)])

    def map_label(self,x,l=None):
        """
        return the MAP labelling of x 
        INPUT:
        - x array of shape (nbitem,dim)
        the data under study
        - l=None array of shape(nbitem,self.k)
        component-wise likelihood
        if l==None, it is recomputed
        OUTPUT:
        - z: array of shape(nbitem): the resulting MAP labelling
        of the rows of x
        """
        if l== None:
            l = self._Estep(x)
        z = np.argmax(l,1)
        return z

    def estimate(self,x,z=None,niter=100,delta = 1.e-4,verbose=0):
        """
        estimation of self given x
        INPUT:
         - x array of shape (nbitem,dim)
        the data from which the model is estimated
        - z = None: array of shape (nbitem)
        a prior labelling of the data to initialize the computation
        - niter=100: maximal number of iterations in the estimation process
        - delta = 1.e-4: increment of data likelihood at which
        convergence is declared
        - verbose=0:
        verbosity mode
        OUTPUT:
        - bic : an asymptotic approximation of model evidence
        """
        # check that the data is OK
        x = self.check_x(x)
        
        # initialization -> Cmeans
        # alternation of E/M step until convergence
        tiny = 1.e-15
        cc = np.zeros(np.shape(self.means))
        nc = np.var(self.means)
        allOld = -np.infty
        for i in range(niter):
            #if np.var(cc-self.means)<delta*nc:
            #    break
            cc = self.means.copy()
            l = self._Estep(x)
            all = np.mean(np.log(np.maximum( np.sum(l,1),tiny)))
            if all<allOld+delta:
                if verbose:
                    print 'iteration:',i, 'log-likelihood:',all,\
                          'old value:',allOld
                break
            else:
                allOld = all
            if verbose:
                print i, all, self.bic(l)
            self._Mstep(x,l)
            
        return self.bic(l)

    def initialize_and_estimate(self,x,z=None,niter=100,delta = 1.e-4,\
                                ninit=1,verbose=0):
        """
        estimation of self given x
        INPUT:
        ------
        - x array of shape (nbitem,dim)
            the data from which the model is estimated
        - z = None: array of shape (nbitem)
            a prior labelling of the data to initialize the computation
        - niter=100: maximal number of iterations in the estimation process
        - delta = 1.e-4: increment of data likelihood at which
             convergence is declared
        - ninit=1: number of initialization performed
            to reach a good solution
        - verbose=0: verbosity mode

        OUTPUT:
        -------
        - the best model is returned
        """
        bestbic = -np.infty
        bestgmm = GMM(self.k,self.dim,self.prec_type)
        bestgmm.init(x)
        
        for i in range(ninit):
            # initialization -> Cmeans
            self.init(x)

            # alternation of E/M step until convergence
            bic = self.estimate(x,niter=niter,delta=delta,verbose=0)
            if bic>bestbic:
                bestbic= bic
                bestgmm.plugin(self.means,self.precisions,self.weights)
        
        return bestgmm

    def train(self,x,z=None,niter=100,delta = 1.e-4,ninit=1,verbose=0):
        """
        idem initialize_and_estimate
        """
        return self.initialize_and_estimate(x,z,niter,delta,ninit,verbose)

    def test(self,x, tiny = 1.e-15):
        """
        returns the log-likelihood of the mixture for x
        INPUT:
        ------
        - x: array of shape (nbitems,self.dim)
        the data used in the estimation process

        OUPTPUT:
        --------
        - ll: array of shape(nbitems)
        the log-likelihood of the rows of x
        """
        return np.log(np.maximum(self.mixture_likelihood(x),tiny)) 

    
    def show_components(self,x,gd,density=None,nbf = -1):
        """
        Function to plot a GMM -WIP
        Currently, works only in 1D and 2D
        """
        if density==None:
            density = self.mixture_likelihood(gd.make_grid())
                
        if gd.dim==1:
            import matplotlib.pylab as mp
            step = 3.5*np.std(x)/np.exp(np.log(np.size(x))/3)
            bins = max(10,int((x.max()-x.min())/step))

            xmin = 1.1*x.min() - 0.1*x.max()
            xmax = 1.1*x.max() - 0.1*x.min()
            h,c = np.histogram(x, bins, [xmin,xmax], normed=True,new=False)
            offset = (xmax-xmin)/(2*bins)
            c+= offset/2
            grid = gd.make_grid()
            if nbf>-1:
                mp.figure(nbf)
            else:
                mp.figure()
            mp.plot(c+offset,h,linewidth=2)
          
            for k in range (self.k):
                mp.plot(grid,density[:,k],linewidth=2)
            mp.title('Fit of the density with a mixture of Gaussians',
                     fontsize=16)

            legend = ['data']
            for k in range(self.k):
                legend.append('component %d' %(k+1))
            l = mp.legend (tuple(legend))
            for t in l.get_texts(): t.set_fontsize(16)
            a,b = mp.xticks()
            mp.xticks(a,fontsize=16)
            a,b = mp.yticks()
            mp.yticks(a,fontsize=16)

            
            mp.show()

        if gd.dim>1:
            print "not implemented yet"

    def show(self,x,gd,density=None,nbf = -1):
        """
        Function to plot a GMM -WIP
        Currently, works only in 1D and 2D
        """
        if density==None:
            density = self.mixture_likelihood(gd,x)
                
        if gd.dim==1:
            import matplotlib.pylab as mp
            step = 3.5*np.std(x)/np.exp(np.log(np.size(x))/3)
            bins = max(10,(x.max()-x.min())/step)
            xmin = 1.1*x.min() - 0.1*x.max()
            xmax = 1.1*x.max() - 0.1*x.min()
            h,c = np.histogram(x, bins, [xmin,xmax], normed=True)
            offset = (xmax-xmin)/(2*bins)
            grid = gd.make_grid()
            if nbf>-1:
                mp.figure(nbf)
            else:
                mp.figure()
            mp.plot(c+offset,h)
            mp.plot(grid,density)
            mp.show()

        if gd.dim==2:
            import matplotlib.pylab as mp
            if nbf>-1:
                mp.figure(nbf)
            else:
                mp.figure()
            xm = gd.lim[0]
            xM = gd.lim[1]
            ym = gd.lim[2]
            yM = gd.lim[3]

            gd0 = gd.nbs[0]
            gd1 = gd.nbs[1]
            Pdens= np.reshape(density,(gd0,np.size(density)/gd0))
            mp.imshow(Pdens.T,None,None,None,'nearest',
                      1.0,None,None,'lower',[xm,xM,ym,yM])
            mp.plot(x[:,0],x[:,1],'.k')
            mp.axis([xm,xM,ym,yM])
            mp.show()
 
    
class GMM_old(GMM):
    """
    This is the old basic GMM class --
    it uses C code and potrentially more efficient storage
    than the standard GMM class,
    so that it can be better suited for very large datasets.
    However, the standard class is more robust
    and should be preferred in general
    
    caveat:
    - GMM_old.precisions has shape (self.k, self.dim**2)
    -> a reshape is needed
    """
    
    def optimize_with_bic(self,data, kvals=None, maxiter = 300,
                          delta = 0.001, ninit=1,verbose = 0):
        """
        Find the optimal GMM using bic criterion.
        The method is run with all the values in kmax for k

        INPUT:
        ------
        data : (n,p) feature array, n = nb items, p=feature dimension
        kvals=None : range of values for k.
            if kvals==None, self.k is used
        maxiter=300 : max number of iterations of the EM algorithm
        delta = 0.001 : criterion on the log-likelihood
            increments to declare converegence
        ninit=1 : number of possible iterations of the GMM estimation
        verbsose=0: verbosity mode

        OUTPUT:
        -------
        Labels : array of shape(n), type np.int,
            discrete labelling of the data items into clusters
        LL : array of shape(n): log-likelihood of the data
        bic : (float) associated bic criterion
        """
        data = self.check_x(data)
        if kvals==None:
            LogLike, Labels, bic = self.estimate(data,None, maxiter,\
                                                 delta, ninit)
            return Labels, LogLike, self.bic(LogLike)
     
        bic_ref = -np.infty
        for k in kvals:
            self.k = k
            nit = 10
            label = np.zeros(data.shape[0])
            mean,label,J = fc.cmeans(data,k,label,nit)            
            Lab,LL, bic = self.estimate(data,label, maxiter, delta, ninit)
            
            if bic>bic_ref:
                kopt = k
                C = self.means.copy()
                P = self.precisions.copy()
                W = self.weights.copy()
                bic_ref = bic
            if verbose:
                print k,LL,bic,kopt
            
        self.means = C
        self.precisions = P
        self.weights = W
        self.k = kopt
        
        if self.prec_type=='full':
            precisions = np.reshape(self.precisions,(self.k,self.dim*self.dim))
        else:
            precisions = self.precisions
        Labels, LogLike  = fc.gmm_partition(data,self.means,precisions,\
                                            self.weights)

        return Labels, LogLike, self.bic_from_ll(LogLike)

    def estimate(self, data, Labels=None, maxiter = 300, delta = 0.001,
                 ninit=1):
        """
        Estimation of the GMM based on data and an EM algorithm

        INPUT:
        ------
        data : (n*p) feature array, n = nb items, p=feature dimension
        Labels=None : prior labelling of the data
            (this may improve convergence)
        maxiter=300 : max number of iterations of the EM algorithm
        delta = 0.001 : criterion on the log-likelihood
            increments to declare converegence
        ninit=1 : number of possible iterations of the GMM estimation

        OUTPUT:
        -------
        Labels : array of shape(n), type np.int:
            discrete labelling of the data items into clusters
        LL : (float) average log-likelihood of the data
        bic : (float) associated bic criterion
        """
        data = self.check_x(data)
        if Labels==None:
            Labels = np.zeros(data.shape[0],'i')
            nit = 10
            C,Labels,J = fc.cmeans(data,self.k,Labels,nit)
        if (self.k>data.shape[0]-1):
            print "too many clusters"
            self.k = data.shape[0]-1

        if self.prec_type=='full':prec_type=0
        if self.prec_type=='diag': prec_type=1
        
        C, P, W, Labels, bll = fc.gmm(data,self.k,Labels,prec_type,
                                     maxiter,delta)
        self.means = C
        if self.prec_type=='diag':
            self.precisions = P
        if self.prec_type=='full':
            self.precisions = np.reshape(P,(self.k,self.dim,self.dim))
        self.weights = W
        self.check()
        
        for i in range(ninit-1):
            Labels = np.zeros(data.shape[0])
            C, P, W, labels, ll = fc.gmm(data,self.k,Labels,
                                         prec_type,maxiter,delta)
            if ll>bll:
                self.means = C
                if self.prec_type=='diag':
                    self.precisions = P
                if self.prec_type=='full':
                    self.precisions = np.reshape(P,(self.k,self.dim,self.dim))
                self.weights = W
                self.check()
                bll = ll
                Labels = labels
        return Labels,bll, self.bic_from_all (bll,data.shape[0])


    def partition(self,data):
        """
        Partitioning the data according to the gmm model

        INPUT:
        -----
        data : (n*p) feature array, n = nb items, p=feature dimension

        OUTPUT:
        -------
        - Labels :  array of shape (n): discrete labelling of the data 
        - LL : array of shape (n): log-likelihood of the data
        - bic : the bic criterion for this model
        """
        data = self.check_x(data)

        if self.prec_type=='full':
            precisions = np.reshape(self.precisions,(self.k,self.dim*self.dim))
        else:
            precisions = self.precisions
        Labels, LogLike  = fc.gmm_partition\
                           (data,self.means,precisions, self.weights)

        return Labels, LogLike, self.bic_from_ll(LogLike)
        
    def test(self,data):
        """
        Evaluating the GMM on some new data

        INPUT
        -----
        data : (n*p) feature array, n = nb items, p=feature dimension

        OUTPUT
        ------
        LL : array of shape (n): the log-likelihood of the data
        """
        data = self.check_x(data)
        if self.prec_type=='full':
            precisions = np.reshape(self.precisions,(self.k,self.dim*self.dim))
        else:
            precisions = self.precisions
            
        Labels, LogLike  = fc.gmm_partition\
                           (data, self.means,precisions, self.weights)
        return LogLike
        

    def sample(self,gd,x,verbose=0):
        """
        Evaluating the GMM on some new data
        INPUT
        data : (n*p) feature array, n = nb items, p=feature dimension
        OUTPUT
        LL : array of shape (n) log-likelihood of the data
        """
        data = gd.make_grid()
        if self.prec_type=='full':
            precisions = np.reshape(self.precisions,(self.k,self.dim*self.dim))
        else:
            precisions = self.precisions
            
        Labels, LogLike  = fc.gmm_partition(\
            data,self.means,precisions, self.weights)
        if verbose:
            self.show(x,gd,np.exp(LogLike))
        return LogLike
        
            
    def bic_from_ll(self,sll):
        """
        computation of bic approximation of evidence
        INPUT:
        - log-likelihood of the data under the model
        OUPUT:
        - the bic value
        """
        
        # number of parameters
        n = sll.size

        if self.prec_type=='full':
            eta = self.k*(1 + self.dim + (self.dim*self.dim+1)/2)-1
        else:
            eta = self.k*(1 + 2*self.dim )-1
        bicc = np.sum(sll)-np.log(n)*eta
        return bicc

    def bic_from_all(self,all,n,tiny = 1.e-15):
        """
        computation of bic approximation of evidence
        INPUT:
        - all : average log-likelihood of the data under the model
        - n number of data points

        OUPUT:
        - the bic value
        """
        if self.prec_type=='full':
            eta = self.k*(1 + self.dim + (self.dim*self.dim+1)/2)-1
        else:
            eta = self.k*(1 + 2*self.dim )-1
        bicc = n*all-np.log(n)*eta
        return bicc

