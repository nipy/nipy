"""
Gaussian Mixture Model Class:
contains the basic fields and methods of GMMs
the high level functions are/should be binded in C

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

    
class GMM:
    """
    This is the basic GMM class
    GMM.k is the number of components in the mixture
    GMM.dim is the dimension of the data
    GMM.centers is an array that contains all the centers of the components
     shape (GMM.k,GMM.dim)
    GMM.precision is an array that contains all the precision of the components
     its shape varies according to GMM.prec_type
    GMM.prec_type type of the precision matrix
    - O: full coavriance matrix, one for each component. shape = (GMM.k,GMM.dim**2)
    - 1 : diagonal covariance matrix, one for each components. shape = (GMM.k,GMM.dim)
    - 2 : diagonal covariance matrix, the same for all component. shape = (1,GMM.dim)
    GMM.weights contains the weights of the components in the mixture
    GMM.estimated is a binary variable that indicates whether the model has been instantiated or not
    """
    
    def __init__(self, k=1, dim = 1, prec_type=1,centers = None, precision = None, weights = None):
        self.k = k
        self.dim = dim
        self.prec_type = prec_type
        self.centers = centers
        self.precision = precision
        self.weights = weights
        self.estimated = 1
        if centers==None:
            self.estimated = 0
        if self.estimated:
            self.check()

    def set_k(self,k):
        """
        To set the value of k
        """
        self.k = k
        
    def check(self):
        """
        Checking the shape of sifferent matrices involved in the model
        """
        if self.centers.shape[0] != self.k:
            raise ValueError," self.centers does not have correct dimensions"
            
        if self.centers.shape[1] != self.dim:
            raise ValueError," self.centers does not have correct dimensions"

        if self.weights.size != self.k:
            raise ValueError," self.eights does not have correct dimensions"

        if self.prec_type==0:
            if self.dim**2 !=  self.precision.shape[1]:
                raise ValueError, "self.precision does not have correct dimensions"
            if self.precision.shape[0] != self.k:
                raise ValueError,"self.precision does not have correct dimensions"

        if self.prec_type==1:
            if self.dim !=  self.precision.shape[1]:
                raise ValueError,"self.precision does not have correct dimensions"
            if self.precision.shape[0] != self.k:
                raise ValueError, "self.precision does not have correct dimensions"
                
        if self.prec_type==2:
            if self.dim !=  self.precision.shape[1]:
                raise ValueError, "self.precision does not have correct dimensions"
            if self.precision.shape[0] != 1:
                raise ValueError, "self.precision does not have correct dimensions"
        
    def check_data(self,data):
        """
        Checking that the data is in correct format
        """
        if self.dim==1:
            data = np.reshape(data,(np.size(data),1))
        if len(data.shape)<2:
            raise ValueError, "Incorrect size for data"
        if data.shape[1]!=self.dim:
            raise ValueError, "Incorrect size for data"
        return data
            

    def assess_divergence(self,data):
        """
        Assess the possible divergence of the algorithm e.g.
        null variance
        div = self.divergence()
        """
        data = self.check_data(data)
        div = False
        if self.estimated:
            # get the minimal distance bteween data points
            from nipy.neurospin.graph import WeightedGraph as WG
            g = WG(data.shape[0])
            g.knn(data,1)
            d = g.weights.min()
            h = d**2/2               

            # get the model of the variance
            if self.prec_type==1:
                svar = 1.0/np.sum(self.precision,0)
                
            if self.prec_type==0:
                for k in range(self.k):
                    from numpy.linalg import eig
                    prec = np.reshape(self.precision[k,:],(self.dim,self.dim))
                    w,v = eig(prec)
                    svar = w.min()

            msvar = svar.min()
            if msvar<h:
                print self.precision
                div = msvar<h
                print msvar,h
        return div

    def optimize_with_BIC(self,data, kvals=None, maxiter = 300, delta = 0.001, ninit=1,verbose = 0):
        """
        Find the optimal GMM using BIC criterion.
        The method is run with all the values in kmax for k
        INPUT
        data : (n*p) feature array, n = nb items, p=feature dimension
        kvals=None : range of values for k.
        maxiter=300 : max number of iterations of the EM algorithm
        delta = 0.001 : criterion on the log-likelihood increments to declare converegence
        ninit=1 : number of possible iterations of the GMM estimation
        verbsose=0: verbosity mode
        OUTPUT
        Labels : (n) array of type ('i') discrete labelling of the data items into clusters
        LL : (float) average log-likelihood of the data
        BIC : (float) associated BIC criterion
        """
        data = self.check_data(data)
        if kvals==None:
            LogLike, Labels, bic = self.estimate(data,None, maxiter, delta, ninit)
        else:
            bic_ref = -np.infty
            for k in kvals:
                self.k = k
                nit = 10
                label = np.zeros(data.shape[0])
                mean,label,J = fc.cmeans(data,k,label,nit)
                            
                Lab,LL, bic = self.estimate(data,label, maxiter, delta, ninit)
                #print k,LL,bic
                self.assess_divergence(data)

                if bic>bic_ref:
                    kopt = k
                    C = self.centers.copy()
                    P = self.precision.copy()
                    W = self.weights.copy()
                    bic_ref = bic
                if verbose:
                    print k,LL,bic,kopt

            
            self.centers = C
            self.precision = P
            self.weights = W
            self.k = kopt
            self.estimated = 1
            Labels, LogLike  = fc.gmm_partition(data,self.centers,self.precision, self.weights)

            #print self.k, " Loglike: ", LogLike
            #print P
            
        return Labels, LogLike, self.BIC(LogLike.mean(),data.shape[0])

    def estimate(self, data, Labels=None, maxiter = 300, delta = 0.001, ninit=1):
        """
        Estimation of the GMM based on data and an EM algorithm
        INPUT
        data : (n*p) feature array, n = nb items, p=feature dimension
        Labels=None : prior labelling of the data (this may improve convergence)
        maxiter=300 : max number of iterations of the EM algorithm
        delta = 0.001 : criterion on the log-likelihood increments to declare converegence
        ninit=1 : number of possible iterations of the GMM estimation
        OUTPUT
        Labels : (n) array of type ('i') discrete labelling of the data items into clusters
        LL : (float) average log-likelihood of the data
        BIC : (float) associated BIC criterion
        """
        data = self.check_data(data)
        if Labels==None:
            Labels = np.zeros(data.shape[0],'i')
            nit = 10
            C,Labels,J = fc.cmeans(data,self.k,Labels,nit)
        if (self.k>data.shape[0]-1):
            print "too many clusters"
            self.k = data.shape[0]-1

        C, P, W, Labels, LL = fc.gmm(data,self.k,Labels,self.prec_type,maxiter,delta)
        self.centers = C
        self.precision = P
        self.weights = W
        self.estimated = 1
        self.check()
        for i in range(ninit-1):
            Labels = np.zeros(data.shape[0])
            C, P, W, labels, ll = fc.gmm(data,self.k,Labels,self.prec_type,maxiter,delta)
            if ll>LL:
                self.centers = C
                self.precision = P
                self.weights = W
                self.estimated = 1
                self.check()
                LL = ll
                Labels = labels
        return Labels,LL, self.BIC(LL,data.shape[0])


    def partition(self,data):
        """
        Partitioning the data according to the gmm model
        INPUT
        data : (n*p) feature array, n = nb items, p=feature dimension
        OUTPUT
        Labels : (n) array of type ('i') discrete labelling of the data items into clusters
        LL : (n) array of type ('d') log-likelihood of the data
        """
        data = self.check_data(data)
        if (self.estimated):
            Labels, LogLike  = fc.gmm_partition(data,self.centers,self.precision, self.weights)
            return Labels, LogLike, self.BIC(LogLike.mean(),data.shape[0])
        else:
            print "the GMM has not been instantiated"
        

    def test(self,data):
        """
        Evaluating the GMM on some new data
        INPUT
        data : (n*p) feature array, n = nb items, p=feature dimension
        OUTPUT

        LL : (n) array of type ('d') log-likelihood of the data
        """
        data = self.check_data(data)
        if (self.estimated):
            Labels, LogLike  = fc.gmm_partition(data,self.centers,self.precision, self.weights)
            return LogLike
        else:
            print "the GMM has not been instantiated"

    def sample(self,gd,X,verbose=0):
        """
        Evaluating the GMM on some new data
        INPUT
        data : (n*p) feature array, n = nb items, p=feature dimension
        OUTPUT
        Labels : (n) array of type ('i') discrete labelling of the data items into clusters
        LL : (n) array of type ('d') log-likelihood of the data
        """
        if (self.estimated):
            data = gd.make_grid()
            Labels, LogLike  = fc.gmm_partition(data,self.centers,self.precision, self.weights)
            if verbose:
                self.show(X,gd,np.exp(LogLike))
            return LogLike
        else:
            print "the GMM has not been instantiated"
            
    def BIC(self, LL,n):
        """
        Computing the value of the BIC critrion of the current GMM,
        given its average log-likelihood LL
        """
        k = float(self.k)
        n = float(n)
        dim = float(self.dim)

        if self.prec_type==0:
            eta = k-1 + k*dim*(dim+3)/2

        if self.prec_type==1:
            eta = k-1 + k*dim*2

        if self.prec_type==2:
            eta = k-1 + (k+1)*dim
        
        BIC = LL-eta*np.log(n)/n
        return BIC

    def show_components(self,X,gd,density=None,nbf = -1):
        """
        Function to plot a GMM -WIP
        Currently, works only in 1D and 2D
        """
        if density==None:
            density = np.exp(self.sample(gd,X))
                
        if gd.dim==1:
            import matplotlib.pylab as mp
            step = 3.5*np.std(X)/np.exp(np.log(np.size(X))/3)
            bins = max(10,(X.max()-X.min())/step)

            xmin = 1.1*X.min() - 0.1*X.max()
            xmax = 1.1*X.max() - 0.1*X.min()
            h,c = np.histogram(X, bins, [xmin,xmax], normed=True)
            offset = (xmax-xmin)/(2*bins)
            grid = gd.make_grid()
            if nbf>-1:
                mp.figure(nbf)
            else:
                mp.figure()
            mp.plot(c+offset,h,linewidth=2)
          
            for k in range (self.k):
                mp.plot(grid,density[:,k],linewidth=2)
            mp.title('Fit of the density with a mixture of Gaussians',fontsize=16)

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

    def show(self,X,gd,density=None,nbf = -1):
        """
        Function to plot a GMM -WIP
        Currently, works only in 1D and 2D
        """
        if density==None:
            density = np.exp(self.sample(gd,X))
                
        if gd.dim==1:
            import matplotlib.pylab as mp
            step = 3.5*np.std(X)/np.exp(np.log(np.size(X))/3)
            bins = max(10,(X.max()-X.min())/step)
            xmin = 1.1*X.min() - 0.1*X.max()
            xmax = 1.1*X.max() - 0.1*X.min()
            h,c = np.histogram(X, bins, [xmin,xmax], normed=True)
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
            #Pdens= np.reshape(density,(gd0,gd1))
            mp.imshow(np.transpose(Pdens),None,None,None,'nearest',1.0,None,None,'lower',[xm,xM,ym,yM])
            mp.plot(X[:,0],X[:,1],'.k')
            mp.axis([xm,xM,ym,yM])
            mp.show()


class BGMM(GMM):

    """
    This class implements Bayesian diagonal GMMs (prec_type = 1)
    Besides the standard fiels of GMMs,
    this class contains the follwing fields
    - prior_centers : array of shape (k,dim):
    the prior on the components means
    - prior_precision : array of shape (k,dim):
    the prior on the components precisions
    - prior_dof : array of shape (k):
    the prior on the dof (should be at least equal to dim)
    - prior_mean_scale : array of shape (k):
    scaling factor of the prior precision on the mean
    - prior_weights  : array of shape (k)
    the prior on the components weights
    - mean_scale : array of shape (k):
    scaling factor of the posterior precision on the mean
    - dof : array of shape (k): the posterior dofs
    """

    def set_priors(self,prior_centers = None, prior_weights = None, prior_precision = None, prior_dof = None,prior_mean_scale = None ):
        """
        Set the prior of the BGMM
        """
        self.prior_centers = prior_centers
        self.prior_weights = prior_weights
        self.prior_precision = prior_precision
        self.prior_dof = prior_dof
        self.prior_mean_scale = prior_mean_scale
        self.prec_type = 1
        self.check_priors()

    def check_priors(self):
        """
        Check that the meain fields have correct dimensions
        """
        if self.prior_centers.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_centers"
        if self.prior_centers.shape[1]!=self.dim:
            raise ValueError,"Incorrect dimension for self.prior_centers"
        if self.prior_precision.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_precision"
        if self.prior_precision.shape[1]!=self.dim:
            raise ValueError,"Incorrect dimension for self.prior_precision"
        if self.prior_dof.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_dof"
        if self.prior_weights.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_weights"
        
        
    def set_empirical_priors(self,X):
        """
        Set the prior in a natural (almost uninformative) fashion given a dataset X
        INPUT:
        - the BGMM priors
        """
        X = self.check_data(X)
        self.prior_dof = self.dim*np.ones(self.k)
        self.prior_weights = 1./self.k*np.ones(self.k)
        self.prior_mean_scale = np.ones(self.k)
        self.prior_centers = np.repeat(np.reshape(X.mean(0),(1,self.dim)),self.k,0)
        self.prior_precision = np.repeat(np.reshape(1./X.var(0),(1,self.dim)),self.k,0)

    def VB_estimate(self,X,niter = 100,delta = 0.0001):
        """
        Estimation of the BGMM using a Variational Bayes approach
        INPUT:
        - X array of shape (nbitems,dim) the input data
        - niter = 100, the maximal number of iterations of the VB algo
        - delta = 0.0001, the increment in log-likelihood to declare convergence 
        OUTPUT:
        - label: array of shape nbitems: resulting MAP labelling
        """
        X = self.check_data(X)

        # pre_cluster the data (this improves convergence...)
        label = np.zeros(X.shape[0])
        nit = 10
        mean,label,J = fc.cmeans(X,self.k,label,nit)

        label, mean, meansc, prec, we,dof,Li = fc.bayesian_gmm (X,self.prior_centers,self.prior_precision,self.prior_mean_scale,self.prior_weights, self.prior_dof,label,niter,delta)
        self.estimated = 1
        self.centers = mean
        self.mean_scale = meansc
        self.precisions = prec
        self.weights = we
        self.dof = dof
        return label

    def VB_sample(self,gd,X=None):
        """
        Sampling of the BGMM model on test points (the 'grid')in order to have
        an estimate of the posterior on these points
        INPUT:
        - gd = a grid descriptor, i.e. 
        the grid on chich the BGMM is sampled
        - X = None: used for plotting (empirical data)
        OUTPUT:
        - Li : array of shape (nbnodes,self.k): the posterior for each node and component
        """
        if self.estimated:
            grid = gd.make_grid()
            Li = fc.bayesian_gmm_sampling(self.prior_centers,self.prior_precision,self.prior_mean_scale,self.prior_weights, self.prior_dof,self.centers,self.precisions,self.mean_scale,self.weights, self.dof,grid)
        else:
            raise ValueError, "the model has not been estimated"

        if X!=None:
            self.show(X,gd,np.exp(Li))
        return Li.sum(1)




    def VB_estimate_and_sample(self,X,niter = 1000,delta = 0.0001,gd = None,verbose = 0):
        """
        Estimation of the BGMM using a Variational Bayes approach,
        and sampling of the model on test points in order to have
        an estimate of the posterior on these points
        INPUT:
        - X array of shape (nbitems,dim) the input data
        - niter = 100, the maximal number of iterations of the VB algo
        - delta = 0.0001, the increment in log-likelihood to declare convergence 
        - gd = None  a grid descriptor, i.e. 
        the grid on chich the model is sampled
        if gd==None, X is used as Grid
        - verbose = 0: the verbosity mode
        OUTPUT:
        - Li : array of shape (nbnodes): the average log-posterior
        - label: array of shape nbitems: resulting MAP labelling
        """
        X = self.check_data(X)
        
        # pre_cluster the data (this improves convergence...)
        label = np.zeros(X.shape[0])
        nit = 10
        mean,label,J = fc.cmeans(X,self.k,label,nit)

        if gd==None:
            grid = X
        else:
            grid = gd.make_grid()
        
        label, mean, meansc, prec, we,dof,Li = fc.bayesian_gmm (X,self.prior_centers,self.prior_precision,self.prior_mean_scale,self.prior_weights, self.prior_dof,label,niter,delta,grid)
        self.estimated = 1
        self.centers = mean
        self.mean_scale = meansc
        self.precisions = prec
        self.weights = we
        self.dof = dof
        if verbose:
            self.show(X,gd,np.exp(Li))
        return Li,label

    def sample_on_data(self,grid):
        """
        Sampling of the BGMM model on test points (the 'grid')in order to have
        an estimate of the posterior on these points
        INPUT:
        - grid: a set of points from which the posterior should be smapled 
        OUTPUT:
        - Li : array of shape (nbnodes,self.k): the posterior for each node and component
        """
        if self.estimated:
            Li = fc.bayesian_gmm_sampling(self.prior_centers,self.prior_precision,self.prior_mean_scale,self.prior_weights, self.prior_dof,self.centers,self.precisions,self.mean_scale,self.weights, self.dof,grid)
        else:
            raise ValueError, "the model has not been estimated"

        return Li

    def Gibbs_estimate(self,X,niter = 1000,method = 1):
        """
        Estimation of the BGMM using Gibbs sampling
        INPUT:
        - X array of shape (nbitems,dim) the input data
        - niter = 1000, the maximal number of iterations of the Gibbs sampling
        - method = 1: boolean to state whether covariance
        are fixed (0 ; normal model) or variable (1 ; normal-wishart model)
        OUTPUT:
        - label: array of shape nbitems: resulting MAP labelling
        """
        X = self.check_data(X)
        label, mean, meansc, prec, we,dof,Li = fc.gibbs_gmm (X,self.prior_centers,self.prior_precision,self.prior_mean_scale,self.prior_weights, self.prior_dof,niter,method)
        self.estimated = 1
        self.centers = mean
        self.mean_scale = meansc
        self.precisions = prec
        self.weights = we
        self.dof = dof
        label = np.argmax(label,1)
        return label
        
    def Gibbs_estimate_and_sample(self,X,niter = 1000,method = 1,gd = None,nsamp = 1000,verbose=0):
        """
        Estimation of the BGMM using Gibbs sampling
        and sampling of the posterior on test points
        INPUT:
        - X array of shape (nbitems,dim) the input data
        - niter = 1000, the maximal number of iterations of the Gibbs sampling
        - method = 1: boolean to state whether covariance
        are fixed (0 ; normal model) or variable (1 ; normal-wishart model)
        - gd = None,  a grid descriptor, i.e. 
        the grid on chich the model is sampled
        if gd==None, X is used as Grid
        - nsamp = 1000 number of draws of the posterior
        -verbose = 0: the verboseity level
        OUTPUT:
        - Li : array of shape (nbnodes): the average log-posterior
        - label: array of shape (nbitems): resulting MAP labelling
        """
        X = self.check_data(X)
        if gd==None:
            grid = X
        else:
            grid = gd.make_grid()
            
        label, mean, meansc, prec, we,dof,Li = fc.gibbs_gmm (X,self.prior_centers,self.prior_precision,self.prior_mean_scale,self.prior_weights, self.prior_dof,niter,method,grid,nsamp)
        self.estimated = 1
        self.centers = mean
        self.mean_scale = meansc
        self.precisions = prec
        self.weights = we
        self.dof = dof
        if verbose:
            self.show(X,gd,Li)

        label = np.argmax(label,1)
        return np.log(Li),label

