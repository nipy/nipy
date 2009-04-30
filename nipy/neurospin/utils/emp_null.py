"""
this module contains a class that fits a gaussian model to the central
part of an histogram, following schwartzman et al, 2009. This is
typically necessary to estimate a fdr when one is not certain that the
data behaves as a standard normal under H_0.

Author : Bertrand Thirion, 2008-2009
"""
import numpy as np
from numpy.random import randn
from numpy.linalg import pinv
import scipy.stats as st

class FDR:
    """
    This is the basic class to handle false discovery rate computation
    parameter:
    fdr.x the samples from which the fdr is derived
    x is assumed to be a normal variate

    The Benjamini-Horchberg procedure is used 
    """
    
    def __init__(self, x):
        """
        x is assumed to be a 1-d array
        """
        self.x = np.squeeze(x)
        
    def all_fdr(self, x=None, verbose=0):
        """
        Returns all the FDR (false discovery rates) values for the sample x
        
        Parameters
        -----------
        x : ndarray of shape (n)
            The normal variates
        
        Results
        -------
        fdr : ndarray of shape (n)
            The set of all FDRs
        """
        if x==None:x=self.x
        pvals = st.norm.sf(x)
        return(self.all_fdr_from_pvals(pvals,verbose))

    def all_fdr_from_pvals(self, pv, verbose=0):
        """
        Returns the fdr associated with each the values

        Parameters
        -----------
        pv : ndarray of shape (n)
            The samples p-value
        
        Returns
        --------
        q : array of shape(n)
            The corresponding fdrs
        """
        pv = self.check_pv(pv)
        if pv==None:
            pv = self.pv
        n = np.size(pv)
        isx = np.argsort(pv)
        q = np.zeros(n)
        for ip in range(n):
            q[isx[ip]] = np.minimum(1, 
                                np.maximum(n*pv[isx[ip]]/(ip+1), q[isx[ip]]))
            if (ip<n-1):
                q[isx[ip+1]] = q[isx[ip]]
        
        if verbose:
            import matplotlib.pylab as mp
            mp.figure()
            mp.plot(pv, q, '.')
        return q

    def check_pv(self, pv):
        """
        Do some basic checks on the pv array: each value should be within [0,1]
        
        Parameters
        ----------
        pv : array of shape (n)
            The sample p-values

        Returns
        --------
        pv : array of shape (n)
            The sample p-values
        """
        pv = np.squeeze(pv)
        if pv.min()<0:
            print pv.min()
            raise ValueError, "Negative p-values"
        if pv.max()>1:
            print pv.max()
            raise ValueError, "P-values greater than 1!"
        return pv

    def pth_from_pvals(self, pv, alpha=0.05):
        """
        Given a set pv of p-values, returns the critical
        p-value associated with an FDR alpha
        
        Parameters
        -----------
        alpha : float
            The desired FDR significance
        pv : array of shape (n) 
            The samples p-value
        
        Returns
        -------
        pth: float
            The p value corresponding to the FDR alpha
        """
        pv = self.check_pv(pv)
        
        npv = np.size(pv)
        pcorr = alpha/npv
        spv = np.sort(pv)
        ip = 0
        pth = 0.
        while (spv[ip]<pcorr*(ip+1))&(ip<npv):
            pth = spv[ip]
            ip = ip+1
        return pth

    def threshold_from_student(self, df, alpha=0.05, x=None):
        """
        Given an array t of student variates with df dofs, returns the 
        critical p-value associated with alpha.
        
        Parameters
        -----------
        df : float
            The number of degrees of freedom
        alpha : float, optional
            The desired significance
        x : ndarray, optional
            The variate. By default self.x is used
        
        Returns
        --------
        th : float
            The threshold in variate value
        """
        df = float(df)
        if x is None:
            x = self.x
        pvals = st.t.sf(x, df)
        pth = self.pth_from_pvals(pvals, alpha)
        return st.t.isf(pth, df)

    def threshold(self, alpha=0.05, x=None):
        """
        Given an array x of normal variates, this function returns the
        critical p-value associated with alpha.
        x is explicitly assumed to be normal distributed under H_0
        
        Parameters
        -----------
        alpha: float, optional
            The desired significance, by default 0.05
        x : ndarray, optional
            The variate. By default self.x is used

        Returns
        --------
        th : float
            The threshold in variate value
        """
        if x==None:x=self.x
        pvals = st.norm.sf(x)
        pth = self.pth_from_pvals(pvals,alpha)
        return st.norm.isf(pth)
    

class ENN:
    """
    Class to compute the empirical null normal fit to the data.

    The data which is used to estimate the FDR, assuming a gaussian null
    from Schwartzmann et al., NeuroImage 44 (2009) 71--82
    """
    
    def __init__(self, x):
        """
        Initiate an empirical null normal object.

        Parameters
        -----------
        x : 1D ndarray
            The data used to estimate the empirical null.
        """
        x = np.reshape(x,(-1, ))
        self.x = np.sort(x)
        self.n = np.size(x)
        self.learned = 0

    def learn(self, left=0.2, right=0.8):
        """
        Estimate the proportion, mean and variance of a gaussian distribution 
        for a fraction of the data

        Parameters
        -----------
        left : float, optional
            Left cut parameter to prevent fitting non-gaussian data         
        right : float, optional 
            Right cut parameter to prevent fitting non-gaussian data

        Notes
        ------

        This method stores the following attributes:
         * mu = mu
         * p0 = min(1, np.exp(lp0))
         * sqsigma : standard deviation of the estimated normal
           distribution
         * sigma = np.sqrt(sqsigma) : variance of the estimated
           normal distribution
        """
        # take a central subsample of x
        x = self.x[int(self.n*left):int(self.n*right)]
    
        # generate the histogram
        step = 3.5*np.std(self.x)/np.exp(np.log(self.n)/3)
        bins = max(10, (self.x.max() - self.x.min())/step)
        hist, ledge = np.histogram(x, bins=bins)
        step = ledge[1]-ledge[0]
        medge = ledge + 0.5*step
    
        # remove null bins
        whist = hist>0
        hist = hist[whist]
        medge = medge[whist]
        hist = hist.astype('f')

        # fit the histogram
        DMtx = np.ones((3, np.sum(whist)))
        DMtx[1] = medge
        DMtx[2] = medge**2
        coef = np.dot(np.log(hist), pinv(DMtx))
        sqsigma = -1.0/(2*coef[2])
        mu = coef[1]*sqsigma
        lp0 = (coef[0]- np.log(step*self.n) 
                + 0.5*np.log(2*np.pi*sqsigma) + mu**2/(2*sqsigma))
        self.mu = mu
        self.p0 = min(1, np.exp(lp0))
        self.sigma = np.sqrt(sqsigma)
        self.sqsigma = sqsigma

    def fdrcurve(self):
        """
        Returns the fdr associated with any point of self.x
        """
        import scipy.stats as st
        if self.learned==0:
            self.learn()
        efp = ( self.p0*st.norm.sf(self.x, self.mu, self.sigma)
               *self.n/np.arange(self.n,0,-1))
        efp = np.minimum(efp, 1)
        return efp

    def threshold(self, alpha=0.05, verbose=0):
        """
        Compute the threshold correponding to an alpha-level fdr for x

        Parameters
        -----------
        alpha : float, optional
            the chosen false discovery rate threshold.
        verbose : boolean, optional
            the verbosity level, if True a plot is generated.
        
        Results
        --------
        theta: float
            the critical value associated with the provided fdr
        """
        efp = self.fdrcurve()
        if verbose:
            self.plot(efp, alpha)
    
        if efp[-1] > alpha:
            print "the maximal value is %f , the corresponding fdr is %f " \
                    % (self.x[-1], efp[-1])
            return np.infty
        j = np.argmin(efp[::-1] < alpha) + 1
        return 0.5*(self.x[-j] + self.x[-j+1])

    def uncorrected_threshold(self, alpha=0.001, verbose=0):
        """
        Compute the threshold correponding to a specificity alpha for x

        Parameters
        -----------
        alpha : float, optional
            the chosen false discovery rate threshold.
        verbose : boolean, optional
            the verbosity level, if True a plot is generated.
        
        Results
        theta: float
            the critical value associated with the provided p-value
        """
        if self.learned==0:
            self.learn()
        threshold = st.norm.isf(alpha, self.mu, self.sigma)
        if not np.isfinite(threshold):
            threshold = np.inf
        if verbose:
            self.plot()
        return threshold

    def fdr(self,theta):
        """
        given a threshold theta, find the estimated fdr
        """
        import scipy.stats as st
        if self.learned==0:
            self.learn()
        efp = self.p0*st.norm.sf(theta,self.mu,self.sigma)*float(self.n)/np.sum(self.x>theta)
        efp = np.minimum(efp,1)
        return efp

    def plot(self, efp=None, alpha=0.05, bar=1):
        """
        plot the  histogram of x
        
        Parameters
        ------------
        efp : float, optional 
            The empirical fdr (corresponding to x)
            if efp==None, the false positive rate threshod plot is not 
            drawn.
        alpha : float, optional 
            The chosen fdr threshold
        """ 
        if not self.learned:
            self.learn()
        n = np.size(self.x)
        bins = max(10, int(2*np.exp(np.log(n)/3.)))
        hist, ledge = np.histogram(self.x, bins=bins)
        hist = hist.astype('f')/hist.sum()
        step = ledge[1]-ledge[0]
        medge = ledge + 0.5*step
        import scipy.stats as st
        g = self.p0*st.norm.pdf(medge, self.mu, self.sigma) 
        hist /= step
        
        import matplotlib.pylab as mp
        mp.figure()
        if bar:
            # We need to cut ledge to len(hist) to accomodate for pre and
            # post numpy 1.3 hist semantic change.
            mp.bar(ledge[:len(hist)], hist, step)
        else:
            mp.plot(medge[:len(hist)], hist, linewidth=2)
        mp.plot(medge, g, 'r', linewidth=2)
        mp.title('Robust fit of the histogram', fontsize=16)
        l = mp.legend(('empiricall null', 'data'), loc=0)
        for t in l.get_texts():
            t.set_fontsize(16)
        a, b = mp.xticks()
        mp.xticks(a, fontsize=16)
        a, b = mp.yticks()
        mp.yticks(a, fontsize=16)

        if efp != None:
            mp.plot(self.x, np.minimum(alpha, efp), 'k')
    

 
