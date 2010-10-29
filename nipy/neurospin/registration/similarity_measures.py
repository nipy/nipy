import numpy as np 

def nonzero(x):
    return np.maximum(x, 1e-200)


class SimilarityMeasure(object): 
    
    def __init__(self, H): 
        self.H = H
        self.J, self.I = np.indices(H.shape) 

    def loss(self): 
        return np.zeros(self.H.shape)

    def npoints(self): 
        return nonzero(self.H.sum())
        
    def averaged_loss(self): 
        return np.sum(self.H*self.loss())/self.npoints()

    def __call__(self):
        return -self.averaged_loss()
        

class MutualInformation(SimilarityMeasure): 

    def loss(self): 
        L = self.H.copy()/self.npoints()
        LT = L.T
        lI = L.sum(0) 
        lJ = L.sum(1)
        L /= lI
        LT /= lJ
        return -np.log(nonzero(L))


class CorrelationCoefficient(SimilarityMeasure): 
    
    def loss(self): 
        rho2 = self()
        neg_rho2 = np.minimum(1-rho2, TINY) 
        I = (self.I-self.mI)/np.sqrt(nonzero(self.vI))
        J = (self.J-self.mJ)/np.sqrt(nonzero(self.vJ))
        L = rho2*I**2 + rho2*J**2 - 2*self.rho*I*J
        L *= .5/neg_rho2
        L += .5*np.log(neg_rho2)
        return L
    
    def averaged_loss(self): 
        neg_rho2 = np.minimum(1-self(), TINY) 
        return .5*np.log(neg_rho2)

    def __call__(self): 
        npts = self.npoints()
        self.mI = np.sum(self.H*self.I)/npts
        self.mJ = np.sum(self.H*self.J)/npts
        self.vI = np.sum(self.H*(self.I)**2)/npts - self.mI**2
        self.vJ = np.sum(self.H*(self.J)**2)/npts - self.mJ**2
        self.cIJ = np.sum(self.H*self.J*self.I)/npts - self.mI*self.mJ
        self.rho = self.cIJ/np.sqrt(self.vI*self.vJ)
        return self.rho**2


class CorrelationRatio(SimilarityMeasure):         

    def loss(self): 
        rho2 = self()
        print('Sorry, not implemented yet!')
        return 

    def __call__(self):
        self.npts_J = nonzero(np.sum(self.H, 1))
        self.mI_J = np.sum(self.H*self.I, 1)/self.npts_J
        self.vI_J = np.sum(self.H*(self.I)**2, 1)/self.npts_J - self.mI_J**2
        npts = self.npoints()
        hI = np.sum(self.H, 0)
        hJ = np.sum(self.H, 1)
        self.mI = np.sum(hI*self.I[0,:])/npts
        self.vI = np.sum(hI*self.I[0,:]**2)/npts - self.mI**2
        mean_vI_J = np.sum(hJ*self.vI_J)/npts
        return 1.-mean_vI_J/nonzero(self.vI)


class ReverseCorrelationRatio(CorrelationRatio): 
    
    def __init__(self, H): 
        self.H = H.T
        self.J, self.I = np.indices(H.T.shape) 


