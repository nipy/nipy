from scipy.special import gammaln
import numpy as np

class TwoBinomialMixture:
    """
    This is the basic Fitting of a mixture of 2 binomial distributions
    it contains the follwing fields:
    - r0=0.2:the parameter of the first binomial
    - r1=0.8: the parameter of the second binomial
    - lambda=0.9 = the mixture parameter (proportion of the first compoenent)
    Note that all these parameters are within the [0,1] interval
    - verbose = 0 verbosity level
    It is now advised to proceed with the estimation using the EM method
    """
    
    def __init__(self, r0=0.2, r1=0.8, l=0.9, v=0):
        # parameters of the two binomial distributions
        self.r0 = r0
        self.r1 = r1
        # mixture parameter
        self.Lambda = l
        self.verbose = v

    def reset(self, r0=0.2, r1=0.8, l=0.9):
        self.r0 = r0
        self.r1 = r1
        self.Lambda = l

    def parameters(self):
        print "first parameter: ", self.r0, " second parameter: ", self.r1
        print " mixture coefficient: ", self.Lambda
        
    def kappa(self):
        """
        Compute the corefficient kappa to measure the separation of
        the two modes
        """
        tau = self.Lambda*(1-self.r0)+(1-self.Lambda)*(1-self.r1)
        Pc = self.Lambda*tau + (1-self.Lambda)*(1-tau)
        P0 = self.Lambda*(1-self.r0)+(1-self.Lambda)*self.r1      
        kappa = (P0-Pc)/(1-Pc)
        return kappa

    def _bcoef(self, n,p):
        if p==0: return 0
        if p==n: return 0
        if p<0: raise ValueError, "negative value for gamma argument"
        if p>n: raise ValueError, "negative value for gamma argument"
        bc = gammaln(n)-gammaln(p)-gammaln(n-p)
        return bc


    def Estep(self,H):
        """
        E-step of the EM algorithm
        """
        nH = np.size(H)
        Z = np.zeros((nH,2))
        ndraw = nH-1
        LL = 0
        for i in range(nH):
            L0 = np.exp(i*np.log(self.r0)+ (ndraw-i)*np.log(1-self.r0))
            L1 = np.exp(i*np.log(self.r1)+ (ndraw-i)*np.log(1-self.r1))
            L = self.Lambda*L0+(1-self.Lambda)*L1   
            Z[i,0] = self.Lambda*L0/L
            Z[i,1] = (1-self.Lambda)*L1/L
            LL += L*H[i]*np.exp(self._bcoef(ndraw,i))
                               
        LL /= np.sum(H)
        if self.verbose:
            print "LL:",LL
        return Z


    def Mstep(self,H,Z):
        """
        M-step of the EM algorithm
        """
        nH = np.size(H)
        # update r0
        A = np.sum(Z[:,0]*H*np.arange(nH))
        B = np.sum(Z[:,0]*H)*(nH-1)
        self.r0 = A/B
                
        #update r1
        A = np.sum(Z[:,1]*H*np.arange(nH))
        B = np.sum(Z[:,1]*H)*(nH-1)
        self.r1 = A/B
        
        #update lambda
        self.Lambda = np.sum(Z[:,0]*H)/np.sum(H)

    def EMalgo(self, X, xmax, eps=1.e-7, maxiter=100,maxh=100):
        """
        Estimate the parameters of the mixture from the input data
        using an EM algorithm
        
        Parameters
        ----------
        X array of shape (nbitems) 
          a vector of interegers in [0,xmax] range
        xmax: the maximal value of the input variable
        eps = 1.e-7 = parameter to decide convergence: when lambda
            changes by less than this amount, convergence is declared
        maxiter=100 : maximal number of iterations
        """
        if xmax<X.max():
            print "xmax is less than the max of X. I cannot proceed"
        else:
            H = np.array([np.sum(X==i) for i in range(min(int(xmax)+1,maxh))])
            self.EMalgo_from_histo(H,eps,maxiter)

    def EMalgo_from_histo(self,H,eps=1.e-7, maxiter=100):
        """
        Estimate the parameters given an histogram of some data, using
        an EM algorithm
        
        Parameters
        ----------
        H the histogram, i.e. the empirical count of values, whose
          range is given by the length of H (to be padded with zeros
          when necesary)
        eps = 1.e-7 
            parameter to decide convergence: when lambda
            changes by less than this amount, convergence is declared
        maxiter=100
        """
        for i in range(maxiter):
            l0 = self.Lambda
            Z = self.Estep(H)
            self.Mstep(H,Z)
            if (np.absolute(self.Lambda-l0)<eps):
                break
    
    def estimate_parameters_from_histo(self, H, eps=1.e-7, maxiter=100,
                        reset=True):
        """
        Estimate the parameters given an histogram of some data
        using a gradient descent.
        this is strongly discouraged: rather use the EM
        
        Parameters
        -----------
        H : 1D ndarray
            The histogram, i.e. the empirical count of values, whose
            range is given by the length of H (to be padded with zeros
            when necesary)
        eps : float, optional
            Parameter to decide convergence: when lambda changes by 
            less than this amount, convergence is declared
        maxiter : float, optional
            Maximal number of iterations
        reset : boolean, optional
            If reset is True, the previously estimate parameters are
            forgotten before performing new estimation.
        """
        self.reset()
        ll = self.update_parameters_fh(H)
        if self.verbose:
            print ll
        for i in range(maxiter):
            l0 = self.Lambda
            self.update_lambda_fh(H)
            ll = self.update_parameters_fh(H)
            if self.verbose:
                print ll
            if (np.absolute(self.Lambda-l0)<eps):
                break


    def update_parameters_fh(self, H, eps=1.e-8):
        """
        update the binomial parameters given a certain histogram
        Parameters
        ----------
        H array of shape (nbins)
          histogram, i.e. the empirical count of values, whose
          range is given by the length of H (to be padded with zeros
          when necesary)
        eps = 1.e-8 
            quantum parameter to avoid zeros and numerical
            degeneracy of the model
        """
        sH = np.size(H)
        mH = sH-1
        K0 = np.exp(np.arange(sH)*np.log(self.r0)+ \
           (mH-np.arange(sH))*np.log(1-self.r0))
        K1 = np.exp(np.arange(sH)*np.log(self.r1)+ \
           (mH-np.arange(sH))*np.log(1-self.r1))
        ll = self.Lambda * K0 + (1-self.Lambda)*K1;

        # first component of the mixture
        Ha = np.sum(H*(mH-np.arange(sH))*K0/ll)
        Hb = np.sum(H*np.arange(sH)*K0/ll);
        Ht = Ha + Hb
        if ((Ht>0)==0):
            self.r0 = eps
        else:
            self.r0 = np.maximum(eps,np.minimum(1-eps,Hb/Ht))

        # second component of the mixture
        Ha = np.sum(H*(mH-np.arange(sH))*K1/ll)
        Hb = np.sum(H*np.arange(sH)*K1/ll);
        Ht = Ha + Hb
        if ((Ht>0)==0):
            self.r1 = 1-eps
        else:
            self.r1 = np.maximum(eps,np.minimum(1-eps,Hb/Ht))

        return np.sum(H*np.log(ll))/sum(H)

    def update_lambda_fh(self, H, eps=1.e-8, maxiter=100):
        """
        update lambda given the histogram H
        
        Parameters
        ----------
        H array of shape (nbins)
          histogram, i.e. the empirical count of values, whose
          range is given by the length of H (to be padded with zeros
          when necesary)
        eps = 1.e-8 
            quantum parameter to avoid zeros and numerical
            degeneracy of the model
        maxiter = 100: maximum number of iterations
        """
        sH = np.size(H)
        K0 = np.exp(np.arange(sH)*np.log(self.r0)+ \
             (sH-np.arange(sH))*np.log(1-self.r0))
        K1 = np.exp(np.arange(sH)*np.log(self.r1)+ \
             (sH-np.arange(sH))*np.log(1-self.r1))
        dK  = K0-K1
        for i in range(maxiter):
            f = np.sum(H*dK/(self.Lambda*dK+K1))
            df = -np.sum(H*dK**2/(self.Lambda*dK+K1)**2)
            dl = -0.5*f/df
            self.Lambda = np.minimum(1-eps,np.maximum(eps,self.Lambda+dl))
            
    def estimate_parameters(self, X, n_bins=10, eps=1.e-7, maxiter=100):
        """
        Estimate the parameters of the mixture from the input data
        using a gradient descent algorithm this is strongly
        discouraged: rather use the EM
        
        Parameters
        -----------
        X : 1D ndarray
            The data to estimate the binomial mixture from.
        n_bins: integer
            The number of bins used to build the histogram.
        eps: float, optional
            Parameter to decide convergence: when lambda changes 
            by less than this amount, convergence is declared.
        maxiter : integer, optional
            Maximal number of iterations
        """
        # XXX: Use of histogram. Is this good for integers?
        h, _ = np.histogram(X, bins=n_bins)
        self.estimate_parameters_from_histo(h, eps, maxiter)

    
    def show(self,H):
        """
        Display the histogram of the data, together with the mixture model

        Parameters
        ----------
        H : ndarray
            The histogram of the data.
        """
        xmax = np.size(H)
        sH = np.sum(H)
        nH = H.astype(float)/sH

        L = np.zeros(xmax)
        ndraw = xmax-1
        for i in range(xmax):
            L0 = np.exp(self._bcoef(ndraw,i)+i*np.log(self.r0)+ \
                 (ndraw-i)*np.log(1-self.r0))
            L1 = np.exp(self._bcoef(ndraw,i)+i*np.log(self.r1)+ \
                 (ndraw-i)*np.log(1-self.r1))
            L[i] = self.Lambda*L0 + (1-self.Lambda)*L1

        L = L/L.sum()
        import matplotlib.pylab as mp
        mp.figure()
        mp.bar(np.arange(xmax),nH)
        mp.plot(np.arange(xmax)+0.5,L,'k',linewidth=2)
        






