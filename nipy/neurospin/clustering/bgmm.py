"""
Bayesian Gaussian Mixture Model Classes:
contains the basic fields and methods of Bayesian GMMs
the high level functions are/should be binded in C

The base class BGMM relies on an implementation that perfoms Gibbs sampling

A derived class VBGMM uses Variational Bayes inference instead

A third class is introduces to take advnatge of the old C-bindings,
but it is limited to diagonal covariance models

fixme: the docs should be rewritten

Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
import numpy.random as nr
import nipy.neurospin.clustering.clustering as fc
from gmm import GMM

# --------------------------------------------
# --- ancillary functions --------------------
# --------------------------------------------
#fixme : this might be put elsewehere



def dirichlet_eval(w,alpha):
    """
    Evaluate the probability of a certain discrete draw w
    from the Dirichlet density with parameters alpha
    INPUT:
    - w: array of shape (n)
    - alpha: array of shape (n)

    FIXME : check that the dimensions of x and alpha are compatible
    """
    from scipy.special import gammaln
    if np.shape(w)!=np.shape(alpha):
        raise ValueError , "incompatible dimensions"
    loge = np.sum((alpha-1)*np.log(w))
    logb = np.sum(gammaln(alpha))-gammaln(alpha.sum())
    loge-= logb
    return np.exp(loge)

def generate_normals(m,P):
    """
    Generate a Gaussian sample
    with mean m and precision P
    INPUT:
    - m array of shape n: the mean vector
    - P array of shape (n,n): the precision matrix
    OUTPUT:
    - ng : array of shape(n): a draw from the gaussian density
    """
    from numpy.linalg import cholesky,inv
    L = inv(cholesky(P))
    ng = nr.randn(m.shape[0])
    ng = np.dot(ng,L)
    ng += m 
    return ng

def generate_Wishart(n,V):
    """
    Generate a sample from Wishart

    INPUT: 
    ------
    - n (scalar) = the number of degrees of freedom (dofs)
    - V = array of shape (n,n) the scale matrix

    OUTPUT:
    -------
    - W: array of shape (n,n): the Wishart draw
    """
    from numpy.linalg import cholesky
    L = cholesky(V)
    p = V.shape[0]
    A = nr.randn(p,p)
    a = np.array([np.sqrt(nr.chisquare(n-i)) for i in range(p)])
    for i in range(p):
        A[i,i:] = 0
        A[i,i] = a[i]
    R = np.dot(L,A)
    W = np.dot(R,R.T)
    return W

def Wishart_eval(n,V,W):
    """
    Evaluation of the  probability of W under Wishart(n,V)

    INPUT: 
    ------
    - n (scalar) = the number of degrees of freedom (dofs)
    - V = array of shape (n,n) the scale matrix
    - W: array of shape (n,n): the Wishart draw

    OUTPUT:
    -------
    - the density
    """
    # check that shape(V)==shape(W)
    from numpy.linalg import det,pinv
    from scipy.special import gammaln
    p = V.shape[0]
    ldW = np.log(det(W))*(n-p-1)/2
    ltr = - np.trace(np.dot(pinv(V),W))/2
    la = ( n*p*np.log(2) + np.log(det(V))*n )/2
    lg = np.log(np.pi)*p*(p-1)/4
    for j in range(p):
        lg += gammaln((n-j)/2)
    lt = ldW + ltr -la -lg
    return np.exp(lt)

def normal_eval(mu,P,x):
    """
    Probability of x under normal(mu,inv(P))

    INPUT:
    ------
    - mu: array of shape (n): the mean parameter
    - P: array of shape (n,n): the precision matrix 
    - x: array of shape (n): the data to be evaluated

    OUTPUT:
    -------
    - the scalar
    """
    from numpy.linalg import det
    p = np.size(mu)
    mu = np.reshape(mu,(1,p))
    w0 = np.log(det(P/2/np.pi))
    w0/=2               
    x = np.reshape(x,(1,p))
    q = np.dot(np.dot(mu-x,P),np.transpose(mu-x))
    w = w0 - q/2
    L = np.exp(w)
    return np.squeeze(L)
        
def generate_perm(k,nperm=100):
    """
    returns an array of shape(nbperm, k) representing
    the permutations of k elements
    - nperm=100
    if k>5: only nperm random draws are generated

    OUPUT:
    p: array of shape(nperm,k): each row is permutation of k
    """
    if k==1:
        return np.reshape(np.array([0]),(1,1)).astype('i')
    if k<6:
        aux = generate_perm(k-1)
        n = aux.shape[0]
        perm = np.zeros((n*k,k)).astype('i')
        for i in range(k):
            perm[i*n:(i+1)*n,:i] = aux[:,:i]
            perm[i*n:(i+1)*n,i] = k-1
            perm[i*n:(i+1)*n,i+1:] = aux[:,i:]
    else:
        from numpy.random import rand
        perm = np.zeros((nperm,k)).astype('i')
        for i in range(nperm):
            p = np.argsort(rand(k))
            perm[i,:] = p
    return perm

def apply_perm(perm,z):
    """
    Permutation of the values of z
    """
    z0 = perm[z.copy()]
    return z0
    
def multinomial(Likelihood):
    """
    Generate samples form a miltivariate distribution
    INPUT:
    - Likelihood: array of shape (nelements, nclasses):
    likelihood of each element belongin to each class
    each row is assumedt to sum to 1
    One sample is draw from each row, resulting in an array

    OUPUT:
    - z array of shape (nelements): the draws,
    that take values in [0..nclasses-1]
    """
    nvox = Likelihood.shape[0]
    nclasses =  Likelihood.shape[1]
    cuml = np.zeros((nvox,nclasses+1))
    cuml[:,1:] = np.cumsum(Likelihood,1)
    aux = np.random.rand(nvox,1)
    z = np.argmax(aux<cuml,1)-1
    return z

def dkl_gaussian(m1,P1,m2,P2):
    """
    Rteun the KL divergence between gausians with densities
    (m1,P1) and (m2,P2)
    where m = mean and P = precision
    """
    from numpy.linalg import det,pinv
    tiny = 1.e-15
    # fixme:check size
    dim = np.size(m1)
    d1 = max(det(P1),tiny)
    d2 = max(det(P2),tiny)
    dkl = np.log(d1/d2)+ np.trace(np.dot(P2,pinv(P1)))-dim
    dkl += np.dot(np.dot((m1-m2).T,P2),(m1-m2))
    dkl /= 2
    return dkl

def dkl_wishart(a1,B1,a2,B2):
    """
    returns the KL divergence bteween two Wishart distribution of
    parameters (a1,B1) and (a2,B2)
    """
    from scipy.special import psi,gammaln
    from numpy.linalg import det,pinv
    tiny = 1.e-15
    # fixme: check size
    dim = B1.shape[0]
    d1 = max(det(B1),tiny)
    d2 = max(det(B2),tiny)
    lgc = dim*(dim-1)*np.log(np.pi)/4
    lg1 = lgc
    lg2 = lgc
    lw1 = -np.log(d1) + dim*np.log(2)
    lw2 = -np.log(d2) + dim*np.log(2)
    for i in range(dim):
        lg1 += gammaln((a1-i)/2)
        lg2 += gammaln((a2-i)/2)
        lw1 += psi((a1-i)/2)
        lw2 += psi((a2-i)/2)
    lz1 = 0.5*a1*dim*np.log(2)-0.5*a1*np.log(d1)+lg1
    lz2 = 0.5*a2*dim*np.log(2)-0.5*a2*np.log(d2)+lg2
    dkl = (a1-dim-1)*lw1-(a2-dim-1)*lw2-a1*dim
    dkl += a1*np.trace(np.dot(B2,pinv(B1)))
    dkl /=2
    dkl += (lz2-lz1)
    return dkl

def dkl_dirichlet(w1,w2):
    """
    returns the KL divergence between two dirichelt distribution of parameters
    w1 and w2
    """
    # fixme: check size
    dkl = 0
    from scipy.special import gammaln, psi
    dkl = np.sum(gammaln(w2))-np.sum(gammaln(w1))
    dkl += gammaln(np.sum(w1))-gammaln(np.sum(w2))
    dkl += np.sum((w1-w2)*(psi(w1)-psi(np.sum(w1))))
    return dkl



# ----------------------------------------
# --------- main GMM class ---------------
# ----------------------------------------


class BGMM(GMM):
    """
    This class implements Bayesian GMMs 

    this class contains the follwing fields
    - k (int): the number of components in the mixture
    - dim (int): is the dimension of the data
    - means array of shape (k,dim):
    all the means of the components
    - precisions array of shape (k,dim,dim):
    the precisions of the componenets    
    - weights: array of shsape (k) weights of the mixture
     - shrinkage : array of shape (k):
    scaling factor of the posterior precisions on the mean
    - dof : array of shape (k): the posterior dofs
    
    - prior_means : array of shape (k,dim):
    the prior on the components means
    - prior_scale : array of shape (k,dim):
    the prior on the components precisions
    - prior_dof : array of shape (k):
    the prior on the dof (should be at least equal to dim)
    - prior_shrinkage : array of shape (k):
    scaling factor of the prior precisions on the mean
    - prior_weights  : array of shape (k)
    the prior on the components weights
    - shrinkage : array of shape (k):
    scaling factor of the posterior precisions on the mean
    - dof : array of shape (k): the posterior dofs

    fixme :
    - E-step and mstep, inhereitde from GMM, should be overriden/removed ?
    - only 'full' preicsion is supported
    """
    
    def __init__(self, k=1, dim=1, means=None, precisions=None,
                 weights=None, shrinkage=None, dof=None):
        """
        Initialize the structure, at least with the dimensions of the problem
        At most, with what is necessary to compute the likelihood of a point
        under the model
        """
        GMM.__init__(self, k, dim, 'full', means, precisions, weights)
        self.shrinkage = shrinkage
        self.dof = dof

        if self.shrinkage==None:
            self.shrinkage = np.ones(self.k)

        if self.dof==None:
            self.dof = np.ones(self.k)
        
    def check(self):
        """
        Checking the shape of sifferent matrices involved in the model
        """
        GMM.check(self)
        
        if self.prior_means.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_means"
        if self.prior_means.shape[1]!=self.dim:
            raise ValueError,"Incorrect dimension for self.prior_means"
        if self.prior_scale.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_scale"
        if self.prior_scale.shape[1]!=self.dim:
            raise ValueError,"Incorrect dimension for self.prior_scale"
        if self.prior_dof.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_dof"
        if self.prior_weights.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_weights"
        
    def set_priors(self,prior_means, prior_weights,
                   prior_scale, prior_dof, prior_shrinkage ):
        """
        Set the prior of the BGMM

        INPUT:
        ------
        - prior_means: array of shape (self.k,self.dim)
        - prior_weights: array of shape (self.k)
        - prior_scale: array of shape (self.k,self.dim,self.dim)
        - prior_dof: array of shape (self.k)
        - prior_shrinkage: array of shape (self.k)
        """
        self.prior_means = prior_means
        self.prior_weights = prior_weights
        self.prior_scale = prior_scale
        self.prior_dof = prior_dof
        self.prior_shrinkage = prior_shrinkage       
        self.check()

    def guess_priors(self,x):
        """
        Set the priors in order of having them weakly uninformative
        this is from  Fraley and raftery;
        Journal of Classification 24:155-181 (2007)

        INPUT:
        ------
        - x:array of shape (nbitems,self.dim)
        the data used in the estimation process
        """
        # a few parameters
        small = 0.01
        mx = np.reshape(x.mean(0),(1,self.dim))
        dx = x-mx
        vx = np.dot(dx.T,dx)/x.shape[0]
        px = np.reshape(np.diag(1.0/np.diag(vx)),(1,self.dim,self.dim))
        px *= np.exp(2.0/self.dim*np.log(self.k))

        # set the priors
        self.prior_means = np.repeat(mx,self.k,0)
        self.prior_weights = np.ones(self.k)
        self.prior_scale = np.repeat(px,self.k,0)
        self.prior_dof = np.ones(self.k)*(self.dim+2)
        self.prior_shrinkage = np.ones(self.k)*small
        self.weights = np.ones(self.k)*1.0/self.k

        # check that everything is OK
        self.check()

    def initialize(self,x):
        """
        initialize z using a k-means algorithm, then upate the parameters

        INPUT:
        ------
        - x:array of shape (nbitems,self.dim)
        the data used in the estimation process
        """
        if self.k>1:
            cent,z,J = fc.cmeans(x,self.k)
        else:
            z = np.zeros(x.shape[0]).astype('i')
        self.update(x,z)
    
    def pop(self,z):
        """
        compute the population, i.e. the statistics of allocation

        INPUT:
        ------
        - z array of shape (nbitems), type = np.int
        the allocation variable

        OUPUT:
        ------
        - hist : count variable, array shape (self.k)
        """
        hist = np.array([np.sum(z==k) for k in range(self.k)])
        return hist

    def update_weights(self,z):
        """
        Given the allocation vector z, resmaple the weights parameter
        
        INPUT:
        ------
        - z array of shape (nbitems), type = np.int
        the allocation variable
        """
        pop = self.pop(z)
        weights = pop+self.prior_weights
        self.weights = np.random.dirichlet(weights)

    def update_means(self,x,z):
        """
        Given the allocation vector z,
        and the corresponding data x,
        resample the mean

        INPUT:
        ------
        - x array of shape (nbitems,self.dim)
        the data used in the estimation process
        - z array of shape (nbitems), type = np.int
        the corresponding classification
        """
        pop = self.pop(z)
        self.shrinkage = self.prior_shrinkage + pop
        empmeans = np.zeros(np.shape(self.means))
        prior_shrinkage = np.reshape(self.prior_shrinkage,(self.k,1))
        shrinkage = np.reshape(self.shrinkage,(self.k,1))

        for k in range(self.k):
            empmeans[k] = np.sum(x[z==k],0)
                
        means = empmeans + self.prior_means*prior_shrinkage
        means/= shrinkage
        for k in range(self.k):
            self.means[k] = generate_normals(\
                means[k],self.precisions[k]*self.shrinkage[k])
        
    def update_precisions(self,x,z):
        """
        Given the allocation vector z,
        and the corresponding data x,
        resample the precisions

        INPUT:
        ------
        - x array of shape (nbitems,self.dim)
        the data used in the estimation process
        - z array of shape (nbitems), type = np.int
        the corresponding classification
        """
        from numpy.linalg import pinv

        pop = self.pop(z)
        self.dof = self.prior_dof + pop +1

        #computing the empirical covariance
        empmeans = np.zeros(np.shape(self.means))
        for k in range(self.k):
            empmeans[k] = np.sum(x[z==k],0)
 
        rpop = (pop+(pop==0)).astype('f')

        empmeans= (empmeans.T/rpop).T

        empcov = np.zeros(np.shape(self.precisions))
        for k in range(self.k):
            dx = np.reshape(x[z==k]-empmeans[k],(pop[k],self.dim))
            empcov[k] += np.dot(dx.T,dx)
                    
        covariance = np.array([pinv(self.prior_scale[k])
                               for k in range(self.k)])
        covariance += empcov
                        
        dx = np.reshape(empmeans-self.prior_means,(self.k,self.dim,1))
        addcov = np.array([np.dot(dx[k],dx[k].T)
                           for k in range(self.k)])
        prior_shrinkage = np.reshape(self.prior_shrinkage,(self.k,1,1))
        covariance += addcov*prior_shrinkage
                
        scale = np.array([pinv(covariance[k]) for k in range(self.k)])
        for k in range(self.k):
            self.precisions[k] = generate_Wishart(self.dof[k],scale[k])

        
    def update(self,x,z):
        """
        update function (draw a sample of the GMM parameters)

        INPUT:
        ------
        - x array of shape (nbitems,self.dim)
        the data used in the estimation process
        - z array of shape (nbitems), type = np.int
        the corresponding classification
        """
        self.update_weights(z)
        self.update_precisions(x,z)
        self.update_means(x,z)
          
    def sample_indicator(self,L):
        """
        sample the indicator from the likelihood

        INPUT:
        ------
        - l: array of shape (nbitem,self.k)
        component-wise likelihood

        OUTPUT:
        ------
        - z: array of shape(nbitem): a draw of the memmbership variable
        
        """
        tiny = 1+1.e-15
        L = np.transpose(np.transpose(L)/L.sum(1))
        L/=tiny
        z = multinomial(L)
        return z

    def sample(self,x,niter=1,mem=0,verbose=0):
        """
        sample the indicator and parameters

        INPUT:
        ------
        - x array of shape (nbitems,self.dim)
        the data used in the estimation process
        - niter=1 : the number of iterations to perform
        - mem=0:
            if mem, the best values of the parameters are computed
        - verbose=0: verbosity mode

        OUTPUT:
        -------
        - best_weights: array of shape (self.k)
        - best_means: array of shape (self.k,self.dim)
        - best_precisions: array of shape (self.k,self.dim,self.dim) 
        - possibleZ: array of shape (nbitems,niter)
        the z that give the highest posterior to the data is returned first
        """
        self.check_x(x)
        if mem:
            possibleZ =  -np.ones((x.shape[0],niter)).astype(np.int)

        score = -np.infty
        bpz = -np.infty
        Mll = 0
        for i in range(niter):
            L = self.likelihood(x)
            sll = np.mean(np.log(np.sum(L,1)))
            sll += np.log(self.probability_under_prior())
            if sll>score:
                score = sll
                best_weights = self.weights.copy()
                best_means = self.means.copy()
                best_precisions = self.precisions.copy()

            z = self.sample_indicator(L)
            if mem:
                possibleZ[:,i] = z
            puz = sll # to save time
            #puz = self._posterior_under_z(x,z)
            self.update(x,z)
            if puz>bpz:
                ibz = i
                bpz = puz
                
        if mem:
            aux = possibleZ[:,0].copy()
            possibleZ[:,0] = possibleZ[:,ibz].copy()
            possibleZ[:,ibz] = aux
            return best_weights, best_means,best_precisions,possibleZ

    def sample_and_average(self,x,niter=1,verbose=0):
        """
        sample the indicator and parameters
        the average values for weights,means, precisions are returned

        INPUT:
        ------
        - x = array of shape (nbitems,dim)
        the data from which bic is computed
        - niter=1: number of iterations

        OUTPUT:
        -------
        - weights: array of shape (self.k)
        - means: array of shape (self.k,self.dim)
        - precisions:  array of shape (self.k,self.dim,self.dim)
        or (self.k, self.dim)
       these are the average parameters across samplings

       NOTE:
       -----
       - All this makes sense only if no label switching as occurred
       so this is wrong in general (asymptotically)
       - fix: implement a permutation procedure for components identification
        """
        aprec  = np.zeros(np.shape(self.precisions))
        aweights  = np.zeros(np.shape(self.weights))
        ameans  = np.zeros(np.shape(self.means))
        for i in range(niter):
            L = self.likelihood(x)
            z = self.sample_indicator(L)
            self.update(x,z)
            aprec += self.precisions
            aweights += self.weights
            ameans += self.means
        aprec/=niter
        ameans/=niter
        aweights/=niter
        return aweights, ameans, aprec


    def probability_under_prior(self):
        """
        Compute the probability of the current parameters of self
        given the priors
        """
        p0 = 1
        p0 = dirichlet_eval(self.weights,self.prior_weights)
        for k in range(self.k):
            mp = self.precisions[k]*self.prior_shrinkage[k]
            p0 *= normal_eval(self.prior_means[k],mp,self.means[k])
            p0 *= Wishart_eval(self.prior_dof[k],self.prior_scale[k],
                               self.precisions[k])
        return p0

    def conditional_posterior_proba(self,x,z):
        """
        Compute the probability of the current parameters of self
        given x and z

         INPUT:
        ------
        - x= array of shape (nbitems,dim)
        the data from which bic is computed
        - z= array of shape (nbitems), type = np.int
        the corresponding classification
        """
        pop = self.pop(z)

        #0. Compute the empirical means
        empmeans = np.zeros(np.shape(self.means))
        for k in range(self.k):
            empmeans[k] = np.sum(x[z==k],0)
 
        rpop = (pop+(pop==0)).astype(np.float)
        empmeans = (empmeans.T/rpop).T
            
        #1. the precisions
        dof = self.prior_dof + pop +1
        empcov = np.zeros(np.shape(self.precisions))

        for k in range(self.k):
            dx = np.reshape(x[z==k]-empmeans[k],(pop[k],self.dim))
            empcov[k] += np.dot(dx.T,dx)

        from numpy.linalg import pinv
                
        covariance = np.array([pinv(self.prior_scale[k])
                               for k in range(self.k)])
        covariance += empcov
                        
        dx = np.reshape(self.means-self.prior_means,
                        (self.k,self.dim,1))
        addcov = np.array([np.dot(dx[k],dx[k].T) for k in range(self.k)])
        prior_shrinkage = np.reshape(self.prior_shrinkage,(self.k,1,1))
        covariance += addcov*prior_shrinkage
        scale = np.array([pinv(covariance[k]) for k in range(self.k)])
    

        #2. the means
        empmeans= (empmeans.T*rpop).T
        shrinkage = self.prior_shrinkage + pop   
        prior_shrinkage = np.reshape(self.prior_shrinkage,(self.k,1))
        shrinkage = np.reshape(shrinkage,(self.k,1))        
        means = empmeans + self.prior_means*prior_shrinkage
        means/= shrinkage

        #3. the weights
        weights = np.array([np.sum(z==k) for k in range(len(self.weights))])
        weights += self.prior_weights
        
        #4. evaluate the posteriors
        pp = 1
        pp = dirichlet_eval(self.weights,weights)
        for k in range(self.k):
            pp*= Wishart_eval(dof[k],scale[k],self.precisions[k])

        for k in range(self.k):
            #mp = self.precisions[k]*shrinkage[k]
            mp = scale[k]*shrinkage[k]
            pp *= normal_eval(means[k],mp,self.means[k])
        return pp
    
    def evidence(self,x,z,nperm=0,verbose=0):
        """
        See Bfactor(self,x,z,nperm=0,verbose=0)
        """
        return self.Bfactor(self,x,z,nperm,verbose)
    
    def Bfactor(self,x,z,nperm=0,verbose=0):
        """
        Evaluate the Bayes Factor of the current model using Chib's method

        INPUT:
        ------
        - x= array of shape (nbitems,dim)
        the data from which bic is computed
        - z= array of shape (nbitems), type = np.int
        the corresponding classification
        - nperm=0 : the number of permutations to sample
        to model the label switching issue in the computation of the BF
        By default, exhaustive permutations are used
        if nperm>0, this number is used.
        Note that this is simply to save time
        - verbose=0: verbosity mode
        
        OUTPUT:
        -------
        - the computed evidence (Bayes factor)

        NOTE:
        -----
        See: Marginal Likelihood from the Gibbs Output
        Journal article by Siddhartha Chib;
        Journal of the American Statistical Association, Vol. 90, 1995
        """
        niter = z.shape[1]
        p = []
        perm = generate_perm(self.k)
        if nperm>perm.shape[0]:
            nperm = perm.shape[0]
        for i in range(niter):
            if nperm==0:
                for j in range(perm.shape[0]):
                    pz = apply_perm(perm[j],z[:,i])
                    temp = self.conditional_posterior_proba(x,pz)
                    p.append(temp)
            else:
                drand = np.argsort(np.random.rand(perm.shape[0]))[:nperm]
                for j in drand:
                    pz = apply_perm(perm[j],z[:,i])
                    temp = self.conditional_posterior_proba(x,pz)
                    p.append(temp)

        p = np.array(p)
        mp = np.mean(p)
        p0 = self.probability_under_prior()
        L = self.likelihood(x)
        bf = np.log(p0) + np.sum(np.log(np.sum(L,1)))- np.log(mp)
        if verbose:
            print np.log(p0), np.sum(np.log(np.sum(L,1))), np.log(mp)
        return bf

    def _posterior_under_z(self,x,z=None):
        """
        return the integrated likelihood of the data
        Using Wishart-Normal model
        the values are not weighted by the component weights

        deprecated
        """
        from numpy.linalg import det,inv
        from scipy.special import gammaln
        if z==None:
            z = np.zeros(x.shape[0],'i')
        w=0
        for k in range(self.k):
            x = x[z==k,:]
            if (np.size(x)>0):
                n = x.shape[0]
                a = self.prior_dof[k]
                tau = self.prior_shrinkage[k]
                tau /= (n+tau)  
                w0 = - n*np.log(np.pi)*self.dim
                w0 += np.log(tau)*self.dim
                w0/=2               
                for j in range(self.dim):
                    w0 += gammaln((a+n-j)/2)
                    w0 -= gammaln((a-j)/2)

                m = np.reshape(self.prior_means[k],(1,self.dim))
                b = self.prior_scale[k]
                ib = inv(b)
                Q = np.dot(np.transpose(x),x)
                M = np.reshape(np.sum(x,0),(1,self.dim))
                f = ib + Q + n*tau*np.dot(np.transpose(m),m)
                f -=  np.dot(np.transpose(M),M)/(self.prior_shrinkage[k]+n)
                f -= tau* (np.dot(np.transpose(m),M)+np.dot(np.transpose(M),m))
                
                w0 += a*np.log(det(ib))/2
                w0 -= (a+n)*np.log(det(f))/2
                w += w0
        return w

# ---------------------------------------------------------
# --- Variational Bayes inference -------------------------
# ---------------------------------------------------------


class VBGMM(BGMM):
    """
    Particular subcalss of Bayesian GMMs (BGMM)
    that implements Variational bayes estimation of the parameters
    """
    
    def __init__(self, k=1, dim=1, means=None, precisions=None,
                 weights=None, shrinkage=None, dof=None):
        BGMM.__init__(self, k, dim, means, precisions, weights,shrinkage, dof)
        self.scale = self.precisions.copy()
        
    def _Estep(self,x):
        """
        VB-E step
        returns the likelihood of the data for each class

        INPUT:
        ------
        - x array of shape (nbitems,dim)
        the data used in the estimation process
        
        OUTPUT:
        -------
        - l array of shape(nbitem,self.k)
        component-wise likelihood
        
        """
        n = x.shape[0]
        L = np.zeros((n,self.k))
        from scipy.special import psi
        from numpy.linalg import det

        spsi = psi(np.sum(self.weights))
        for k in range(self.k):
            # compute the data-independent factor first
            w0 = psi(self.weights[k])-spsi
            w0 += 0.5*np.log(det(self.scale[k]))
            w0 -= self.dim*0.5/self.shrinkage[k]
            w0 += 0.5*np.log(2)*self.dim
            for i in range (self.dim):
                w0 += 0.5*psi((self.dof[k]-i)/2) 
            m = np.reshape(self.means[k],(1,self.dim))
            b = self.dof[k]*self.scale[k]
            q = np.sum(np.dot(m-x,b)*(m-x),1)
            w = w0 - q/2
            w -= 0.5*np.log(2*np.pi)*self.dim 
            L[:,k] = np.exp(w)   

        if L.min()<0: stop
        return L

    def evidence(self,x,L = None):
        """
        computation of evidence or integrated likelihood

        INPUT:
        ------
        - x array of shape (nbitems,dim)
        the data from which bic is computed
        - l=None: array of shape (nbitem,self.k)
        component-wise likelihood
        If None, it is recomputed
        
        OUTPUT:
        -------
        - the computed evidence
        """
        from scipy.special import psi
        from numpy.linalg import det,pinv
        tiny = 1.e-15
        if L==None:
            L = self._Estep(x)
            L = (L.T/np.maximum(L.sum(1),tiny)).T

        pop = L.sum(0)[:self.k]  
        pop = np.reshape(pop,(self.k,1))
        spsi = psi(np.sum(self.weights))
        empmeans = np.dot(L.T[:self.k],x)/np.maximum(pop,tiny)
                
        F = 0
        # start with the average likelihood term
        for k in range(self.k):
            # compute the data-independent factor first
            Lav = psi(self.weights[k])-spsi
            Lav -= np.sum(L[:,k]*np.log(np.maximum(L[:,k],tiny)))/pop[k]
            Lav -= 0.5*self.dim*np.log(2*np.pi)
            Lav += 0.5*np.log(det(self.scale[k]))
            Lav += 0.5*np.log(2)*self.dim
            for i in range (self.dim):
                Lav += 0.5*psi((self.dof[k]-i)/2)
            Lav -= self.dim*0.5/self.shrinkage[k]
            Lav*= pop[k]
            
            empcov = np.zeros((self.dim,self.dim))
            dx = x-empmeans[k]
            empcov = np.dot(dx.T,L[:,k:k+1]*dx)
            Lav -= 0.5*np.trace(np.dot(empcov,self.scale[k]*self.dof[k]))
            F+= Lav
            
        #then the KL divergences
        prior_covariance = np.array([pinv(self.prior_scale[k])
                               for k in range(self.k)])
        covariance = np.array([pinv(self.scale[k]) for k in range(self.k)])
        Dkl = dkl_dirichlet(self.weights,self.prior_weights)
        for k in range(self.k):
            Dkl += dkl_wishart(self.dof[k],covariance[k],
                               self.prior_dof[k],prior_covariance[k])
            nc = self.scale[k]*(self.dof[k]*self.shrinkage[k])
            nc0 = self.scale[k]*(self.dof[k]*self.prior_shrinkage[k])
            Dkl += dkl_gaussian(self.means[k],nc,self.prior_means[k],nc0)
            
        return F-Dkl

    def _Mstep(self,x,L):
        """
        VB-M step

         INPUT:
        ------
        - x: array of shape(nbitem,self.dim)
        the data from which the model is estimated
        - L: array of shape(nbitem,self.k)
        the likelihood of the data under each class
        """
        from numpy.linalg import pinv
        tiny  =1.e-15
        pop = L.sum(0)
       
        # shrinkage,weights,dof
        self.weights = self.prior_weights + pop
        pop = pop[0:self.k]
        L = L[:,:self.k]
        self.shrinkage = self.prior_shrinkage + pop
        self.dof = self.prior_dof + pop
        
        #reshape
        pop = np.reshape(pop,(self.k,1))
        prior_shrinkage = np.reshape(self.prior_shrinkage,(self.k,1))
        shrinkage = np.reshape(self.shrinkage,(self.k,1))

        # means
        means = np.dot(L.T,x)+ self.prior_means*prior_shrinkage
        self.means= means/shrinkage
        
        #precisions
        empmeans = np.dot(L.T,x)/np.maximum(pop,tiny)
        empcov = np.zeros(np.shape(self.prior_scale))
        for k in range(self.k):
             dx = x-empmeans[k]
             empcov[k] = np.dot(dx.T,L[:,k:k+1]*dx) 
                    
        covariance = np.array([pinv(self.prior_scale[k])
                               for k in range(self.k)])
        covariance += empcov

        dx = np.reshape(empmeans-self.prior_means,(self.k,self.dim,1))
        addcov = np.array([np.dot(dx[k],dx[k].T) for k in range(self.k)])
        apms =  np.reshape(prior_shrinkage*pop/shrinkage,(self.k,1,1))
        covariance += addcov*apms

        self.scale = np.array([pinv(covariance[k]) for k in range(self.k)])

        #compute the map of the precisions
        #(not used, but for completness and interpretation)
        

        
    def initialize(self,x):
        """
        initialize z using a k-means algorithm, then upate the parameters

        INPUT:
        ------
        - x:array of shape (nbitems,self.dim)
        the data used in the estimation process
        """
        n = x.shape[0]
        if self.k>1:
            cent,z,J = fc.cmeans(x,self.k)
        else:
            z = np.zeros(x.shape[0]).astype(np.int)
        L = np.zeros((n,self.k))
        L[np.arange(n),z]=1
        self._Mstep(x,L)

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
            l = self.likelihood(x)
        z = np.argmax(l,1)
        return z   

    def estimate(self,x,niter=100,delta = 1.e-4,verbose=0):
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
        - verbose=0:
        verbosity mode
        """
        # alternation of E/M step until convergence
        tiny = 1.e-15
        cc = np.zeros(np.shape(self.means))
        allOld = -np.infty
        for i in range(niter):
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
            l = (l.T/np.maximum(l.sum(1),tiny)).T
            self._Mstep(x,l)
            
    def likelihood(self,x):
        """
        return the likelihood of the model for the data x
        the values are weighted by the components weights

        INPUT:
        ------
        - x:array of shape (nbitems,self.dim)
        the data used in the estimation process

        OUTPUT:
        ------
        - l array of shape(nbitem,self.k)
        component-wise likelihood
        """
        x = self.check_x(x)
        return self._Estep(x) 
            

def pop(self,l,tiny = 1.e-15):
        """
        compute the population, i.e. the statistics of allocation

        INPUT:
        ------
        - l array of shape (nbitem,self.k):
        the likelihood of each item being in each class
        """
        sl = np.maximum(tiny,np.sum(l,1))
        nl = (l.T/sl).T
        return np.sum(nl,0)


# ------------------------------------------
# ----- C bindings -------------------------
# ------------------------------------------

class BGMM_old(GMM):

    """
    This class implements Bayesian diagonal GMMs (prec_type = 1)
    Besides the standard fiels of GMMs,
    this class contains the follwing fields
    - prior_means : array of shape (k,dim):
    the prior on the components means
    - prior_precisions : array of shape (k,dim):
    the prior on the components precisions
    - prior_dof : array of shape (k):
    the prior on the dof (should be at least equal to dim)
    - prior_shrinkage : array of shape (k):
    scaling factor of the prior precisions on the mean
    - prior_weights  : array of shape (k)
    the prior on the components weights
    - shrinkage : array of shape (k):
    scaling factor of the posterior precisions on the mean
    - dof : array of shape (k): the posterior dofs
    
    fixme : needs renaming
    """

    def set_priors(self,prior_means = None, prior_weights = None, prior_precisions = None, prior_dof = None,prior_shrinkage = None ):
        """
        Set the prior of the BGMM
        """
        self.prior_means = prior_means
        self.prior_weights = prior_weights
        self.prior_precisions = prior_precisions
        self.prior_dof = prior_dof
        self.prior_shrinkage = prior_shrinkage
        self.prec_type = 1
        self.check_priors()

    def check_priors(self):
        """
        Check that the meain fields have correct dimensions
        """
        if self.prior_means.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_means"
        if self.prior_means.shape[1]!=self.dim:
            raise ValueError,"Incorrect dimension for self.prior_means"
        if self.prior_precisions.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_precisions"
        if self.prior_precisions.shape[1]!=self.dim:
            raise ValueError,"Incorrect dimension for self.prior_precisions"
        if self.prior_dof.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_dof"
        if self.prior_weights.shape[0]!=self.k:
            raise ValueError,"Incorrect dimension for self.prior_weights"
        
        
    def set_empirical_priors(self,x):
        """
        Set the prior in a natural (almost uninformative) fashion given a dataset x
        INPUT:
        - the BGMM priors
        """
        x = self.check_data(x)
        self.prior_dof = self.dim*np.ones(self.k)
        self.prior_weights = 1./self.k*np.ones(self.k)
        self.prior_shrinkage = np.ones(self.k)
        self.prior_means = np.repeat(np.reshape(x.mean(0),(1,self.dim)),self.k,0)
        self.prior_precisions = np.repeat(np.reshape(1./x.var(0),(1,self.dim)),self.k,0)

    def VB_estimate(self,x,niter = 100,delta = 0.0001):
        """
        Estimation of the BGMM using a Variational Bayes approach
        INPUT:
        - x array of shape (nbitems,dim) the input data
        - niter = 100, the maximal number of iterations of the VB algo
        - delta = 0.0001, the increment in log-likelihood to declare convergence 
        OUTPUT:
        - label: array of shape nbitems: resulting MAP labelling
        """
        x = self.check_data(x)

        # pre_cluster the data (this improves convergence...)
        label = np.zeros(x.shape[0])
        nit = 10
        mean,label,J = fc.cmeans(x,self.k,label,nit)

        label, mean, meansc, prec, we,dof,Li = fc.bayesian_gmm (x,self.prior_means,self.prior_precisions,self.prior_shrinkage,self.prior_weights, self.prior_dof,label,niter,delta)
        self.estimated = 1
        self.means = mean
        self.shrinkage = meansc
        self.precisions = prec
        self.weights = we
        self.dof = dof
        return label

    def VB_sample(self,gd,x=None):
        """
        Sampling of the BGMM model on test points (the 'grid')in order to have
        an estimate of the posterior on these points
        INPUT:
        - gd = a grid descriptor, i.e. 
        the grid on chich the BGMM is sampled
        - x = None: used for plotting (empirical data)
        OUTPUT:
        - Li : array of shape (nbnodes,self.k): the posterior for each node and component
        """
        if self.estimated:
            grid = gd.make_grid()
            Li = fc.bayesian_gmm_sampling(self.prior_means,self.prior_precisions,self.prior_shrinkage,self.prior_weights, self.prior_dof,self.means,self.precisions,self.shrinkage,self.weights, self.dof,grid)
        else:
            raise ValueError, "the model has not been estimated"

        if x!=None:
            self.show(x,gd,np.exp(Li))
        return Li.sum(1)




    def VB_estimate_and_sample(self,x,niter = 1000,delta = 0.0001,gd = None,verbose = 0):
        """
        Estimation of the BGMM using a Variational Bayes approach,
        and sampling of the model on test points in order to have
        an estimate of the posterior on these points
        INPUT:
        - x array of shape (nbitems,dim) the input data
        - niter = 100, the maximal number of iterations of the VB algo
        - delta = 0.0001, the increment in log-likelihood to declare convergence 
        - gd = None  a grid descriptor, i.e. 
        the grid on chich the model is sampled
        if gd==None, x is used as Grid
        - verbose = 0: the verbosity mode
        OUTPUT:
        - Li : array of shape (nbnodes): the average log-posterior
        - label: array of shape nbitems: resulting MAP labelling
        """
        x = self.check_data(x)
        
        # pre_cluster the data (this improves convergence...)
        label = np.zeros(x.shape[0])
        nit = 10
        mean,label,J = fc.cmeans(x,self.k,label,nit)

        if gd==None:
            grid = x
        else:
            grid = gd.make_grid()
        
        label, mean, meansc, prec, we,dof,Li = fc.bayesian_gmm (x,self.prior_means,self.prior_precisions,self.prior_shrinkage,self.prior_weights, self.prior_dof,label,niter,delta,grid)
        self.estimated = 1
        self.means = mean
        self.shrinkage = meansc
        self.precisions = prec
        self.weights = we
        self.dof = dof
        if verbose:
            self.show(x,gd,np.exp(Li))
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
            Li = fc.bayesian_gmm_sampling(self.prior_means,self.prior_precisions,self.prior_shrinkage,self.prior_weights, self.prior_dof,self.means,self.precisions,self.shrinkage,self.weights, self.dof,grid)
        else:
            raise ValueError, "the model has not been estimated"

        return Li

    def Gibbs_estimate(self,x,niter = 1000,method = 1):
        """
        Estimation of the BGMM using Gibbs sampling
        INPUT:
        - x array of shape (nbitems,dim) the input data
        - niter = 1000, the maximal number of iterations of the Gibbs sampling
        - method = 1: boolean to state whether covariance
        are fixed (0 ; normal model) or variable (1 ; normal-wishart model)
        OUTPUT:
        - label: array of shape nbitems: resulting MAP labelling
        """
        x = self.check_data(x)
        label, mean, meansc, prec, we,dof,Li = fc.gibbs_gmm (x,self.prior_means,self.prior_precisions,self.prior_shrinkage,self.prior_weights, self.prior_dof,niter,method)
        self.estimated = 1
        self.means = mean
        self.shrinkage = meansc
        self.precisions = prec
        self.weights = we
        self.dof = dof
        label = np.argmax(label,1)
        return label
        
    def Gibbs_estimate_and_sample(self,x,niter = 1000,method = 1,gd = None,nsamp = 1000,verbose=0):
        """
        Estimation of the BGMM using Gibbs sampling
        and sampling of the posterior on test points
        INPUT:
        - x array of shape (nbitems,dim) the input data
        - niter = 1000, the maximal number of iterations of the Gibbs sampling
        - method = 1: boolean to state whether covariance
        are fixed (0 ; normal model) or variable (1 ; normal-wishart model)
        - gd = None,  a grid descriptor, i.e. 
        the grid on chich the model is sampled
        if gd==None, x is used as Grid
        - nsamp = 1000 number of draws of the posterior
        -verbose = 0: the verboseity level
        OUTPUT:
        - Li : array of shape (nbnodes): the average log-posterior
        - label: array of shape (nbitems): resulting MAP labelling
        """
        x = self.check_data(x)
        if gd==None:
            grid = x
        else:
            grid = gd.make_grid()
            
        label, mean, meansc, prec, we,dof,Li = fc.gibbs_gmm (x,self.prior_means,self.prior_precisions,self.prior_shrinkage,self.prior_weights, self.prior_dof,niter,method,grid,nsamp)
        self.estimated = 1
        self.means = mean
        self.shrinkage = meansc
        self.precisions = prec
        self.weights = we
        self.dof = dof
        if verbose:
            self.show(x,gd,Li)

        label = np.argmax(label,1)
        return np.log(Li),label

