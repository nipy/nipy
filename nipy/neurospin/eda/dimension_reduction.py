"""
This modules contains several classes to perform non-linear dimension
reduction.
Each class has 2 methods, 'train' and 'test'
- 'train' performs the computation of low-simensional data embedding
and the information to generalize to new data
- 'test' computes the embedding for new dsamples of data
This is done for
- Multi-dimensional scaling
- Isompap (knn or eps-neighb implementation)
- Locality Preseving projections (LPP)
- laplacian embedding (train only)

Future developpements will include some supervised cases, e.g. LDA,LDE
and the estimation of the latent dimension, at least in simple cases.
"""

import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import nipy.neurospin.graph.graph as fg
import nipy.neurospin.graph.field as ff

# ------------------------------------------------------------------
# ------------- Auxiliary functions --------------------------------
# ------------------------------------------------------------------

def _linear_dim_criterion_(l,k,dimf,n):
    """
    likelihood = _linear_dim_criterion_(k,l,dimf,n)
    this function returns the likelihood of a dataset
    with rank k embedded in gaussian noise of shape(n,dimf)
    INPUT:
    - l = spectrum
    - k = test rank
    - dimf = maximal rank
    - n number of inputs
    OUPUT:
    The log-likelihood
    NOTE: This is imlpempented from Minka et al., 2001
    """
    if k>dimf:
        raise ValueError, "the dimension cannot exceed dimf"
    import scipy.special as SP
    Pu = -k*np.log(2)
    for i in range(k):
        Pu += SP.gammaln((dimf-i)/2)-np.log(np.pi)*(dimf-i)/2
        
    pl = np.sum(np.log(l[:k]))
    Pl = -pl*n/2

    if k==dimf:
        Pv = 0
        v=1
    else:
        v = np.sum(l[k:dimf])/(dimf-k)
        Pv = -np.log(v)*n*(dimf-k)/2
    
    m = dimf*k-k*(k+1)/2
    Pp = np.log(2*np.pi)*(m+k+1)/2

    Pa = 0
    nl = l.copy()
    nl[k:dimf] = v
    for i in range(k):
        for j in range (i+1,dimf):
            Pa = Pa + np.log((l[i]-l[j])*(1./nl[j]-1./nl[i]))+np.log(n)

    Pa = -Pa/2
    lE = Pu+Pl+Pv+Pp+Pa-k*np.log(n)/2

    return lE

def infer_latent_dim(X,verbose = 0, maxr = -1):
    """
    r = infer_latent_dim(X,verbose = 0)
    Infer the latent dimension of an aray assuming data+gaussian noise mixture
    INPUT:
    - an array X
    - verbose=0 : verbositry level
    - maxr=-1 maximum dimension that can be achieved
    if maxr = -1, this is equal to rank(X)
    OUPTUT
    - r the inferred dimension
    """
    if maxr ==-1:
        maxr = np.minimum(X.shape[0],X.shape[1])
        
    import numpy.linalg as L
    U,S,V = nl.svd(X,0)
    if verbose>1:
        print "Singular Values", S
    L = []
    for k in range(maxr):
        L.append(_linear_dim_criterion_(S,k,X.shape[1],X.shape[0])/X.shape[0])

    L = np.array(L)
    rank = np.argmax(L)
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.bar(np.arange(maxr),L-L.mean())
        mp.show()

    return rank


def Euclidian_distance(X,Y=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vector 
    """
    if Y == None:
        Y = X
    if X.shape[1]!=Y.shape[1]:
        raise ValueError, "incompatible dimension for X and Y matrices"
    
    s1 = X.shape[0]
    s2 = Y.shape[0]
    NX = np.reshape(np.sum(X*X,1),(s1,1))
    NY = np.reshape(np.sum(Y*Y,1),(1,s2))
    ED = np.repeat(NX,s2,1)
    ED = ED + np.repeat(NY,s1,0)
    ED = ED-2*np.dot(X,np.transpose(Y))
    ED = np.maximum(ED,0)
    ED = np.sqrt(ED)
    return ED

def CCA(X,Y,eps = 1.e-12):
    """
    Canonical corelation analysis of two matrices
    INPUT:
    - X and Y are (nbitem,p) and (nbitem,q) arrays that are analysed
    - eps=1.e-12 is a small biasing constant
    to grant invertibility of the matrices
    OUTPUT
    - ccs: the canconical correlations
    NOTE
    - It is expected that nbitem>>max(p,q)
    - In general it makes more sense if p=q
    """
    from numpy.linalg import cholesky,inv,svd
    if Y.shape[0]!=X.shape[0]:
        raise ValueError,"Incompatible dimensions for X and Y"
    nb = X.shape[0]
    p = X.shape[1]
    q = Y.shape[1]
    sqX = np.dot(np.transpose(X),X)
    sqY = np.dot(np.transpose(Y),Y)
    sqX += np.trace(sqX)*eps*np.eye(p)
    sqY += np.trace(sqY)*eps*np.eye(q)
    rsqX = cholesky(sqX)# sqX = rsqX*rsQx^T 
    rsqY = cholesky(sqY)
    iX = np.transpose(inv(rsqX))
    iY = np.transpose(inv(rsqY))
    Cxy = np.dot(np.transpose(np.dot(X,iX)),np.dot(Y,iY))
    uv,ccs,vv = svd(Cxy)
    return ccs


def Euclidian_mds(X,dim,verbose=0):
    """
    returns a dim-dimensional MDS representation of the rows of X 
    using an Euclidian metric 
    """
    d = Euclidian_distance(X)
    return(mds(d,dim,verbose))
    

def mds(dg,dim=1,verbose=0):
    """
    Multi-dimensional scaling, i.e. derivation of low dimensional
    representations from distance matrices.
    INPUT:
    - dg: a (nbitem,nbitem) distance matrix
    - dim=1: the dimension of the desired representation
    - verbose=0: verbosity level
    """
    
    # take the square distances and center the matrix
    dg = dg*dg
    rm0 = dg.mean(0)
    rm1 = dg.mean(1)
    mm = dg.mean()
    dg = dg-rm0
    dg = np.transpose(np.transpose(dg)-rm1)
    dg = dg+mm  
    
    import numpy.linalg as L
    U,S,V = nl.svd(dg,0)
    S = np.sqrt(S)
    
    chart = np.dot(U,np.diag(S))
    chart = chart[:,:dim]

    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.bar(np.arange(np.size(S)),S)
        mp.show()
        
    return chart,np.transpose(V),rm0

def isomap_dev(G,dim=1,p=300,verbose = 0):
    """
    chart,proj,offset =isomap(G,dim=1,p=300,verbose = 0)
    Isomapping of the data
    return the dim-dimensional ISOMAP chart that best represents the graph G
    INPUT:
    - G : Weighted graph that represents the data
    - dim=1 : number of dimensions
    - p=300 : nystrom reduction of the problem
    - verbose = 0: verbosity level
    OUTPUT
    - chart, array of shape(G.V,dim)
    NOTE:
    - this 'dev' version is expected to yield more accurate results
    than the other approximation,
    because of a better out of samples generalization procedure.
    """
    n = G.V
    dim = np.minimum(dim,n)
    p = np.minimum(p,n)

    # get the geodesic distances in the graph
    if p<n:
        toto = nr.rand(n)
        seed = toto.argsort()[:p]
        dg = G.floyd(seed)
    else:
        seed = np.arange(n)
        dg = G.floyd()
        
    dg = np.transpose(dg)
    dg1 = dg[seed]
    
    dg1 = dg1*dg1/2
    rm0 = dg1.mean(0)
    rm1 = dg1.mean(1)
    mm = dg1.mean()
    dg1 = dg1-rm0
    dg1 = np.transpose(np.transpose(dg1)-rm1)
    dg1 = dg1+mm    
    import numpy.linalg as L
    U,S,V = nl.svd(dg1,0)
    S = np.sqrt(S)
    chart = np.dot(U,np.diag(S))
    proj = np.transpose(V)

    dg = dg*dg/2
    dg = np.transpose(np.transpose(dg)-np.mean(dg,1))
    Chart = np.dot(np.dot(dg,proj),np.diag(1.0/S))
    return Chart[:,:dim]

def isomap(G,dim=1,p=300,verbose = 0):
    """
    chart,proj,offset =isomap(G,dim=1,p=300,verbose = 0)
    Isomapping of the data
    return the dim-dimensional ISOMAP chart that best represents the graph G
    INPUT:
    - G : Weighted graph that represents the data
    - dim=1 : number of dimensions
    - p=300 : nystrom reduction of the problem
    - verbose = 0: verbosity level
    OUTPUT
    - chart, array of shape(G.V,dim)
    """
    n = G.V
    dim = np.minimum(dim,n)
    p = np.minimum(p,n)

    # get the geodesic distances in the graph
    if p<n:
        toto = nr.rand(n)
        seed = toto.argsort()[:p]
        dg = G.floyd(seed)
    else:
        dg = G.floyd()

    chart,proj,offset = mds(np.transpose(dg),dim,verbose)
    
    return chart,proj,offset



def LE_dev(G,dim,verbose=0,maxiter=1000):
    """
    Laplacian Embedding of the data
    returns the dim-dimensional LE of the graph G
    INPUT:
    - G : Weighted graph that represents the data
    - dim=1 : number of dimensions
    - verbose = 0: verbosity level
    - maxiter=1000: maximum number of iterations of the algorithm 
    OUTPUT
    - chart, array of shape(G.V,dim)
    """
    n = G.V
    dim = np.minimum(dim,n)
    chart = nr.randn(G.V,dim+2)
    f = ff.Field(G.V,G.edges,G.weights,chart)
    S = f.normalize(0)
    eps = 1.e-7

    f1 = np.zeros((G.V,dim+2))
    import numpy.linalg as L
    for i in range(maxiter):
        f.diffusion(10)
        f0 = Orthonormalize(f.field)
        f.field = f0
        if nl.norm(f1-f0)<eps:
            break
        else:
            f1 = f0
    
    if verbose:
        print i,nl.norm(f1-f0)
    f.diffusion()
    f0 = f.field

    LE = np.sqrt(np.sum(f0**2,0))
    U = Orthonormalize(f.field)
    chart = np.transpose(np.transpose(U[:,1:dim+1])/S)
    chart = chart/np.sqrt(np.sum(chart**2,0))


    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.bar(np.arange(np.size(LE)-1),np.sqrt(1-LE[1:]))
        mp.show()
        print 1-LE
        
    return chart


def Orthonormalize(M):
    """
    M = Orthonormalize(M)
    orthonormalize the columns of M
    (Gram-Schmidt procedure)
    """
    C = nl.cholesky(np.dot(np.transpose(M),M))
    M = np.dot(M,np.transpose(nl.inv(C)))
    return M

def local_sym_normalize(G):
    """
    graph symmetric normalization; moiover, the normalizing vector is returned
    NB : this is now in the C graph lib.
    """
    LNorm = np.zeros(G.V)
    RNorm = np.zeros(G.V)
    for e in range(G.E):
        a = G.edges[e,0]
        b = G.edges[e,1]
        d = G.weights[e]
        LNorm[a] = LNorm[a] + d
        RNorm[b] = RNorm[b] + d

    LNorm[LNorm==0]=1
    RNorm[RNorm==0]=1
    
    for e in range(G.E):
        a = G.edges[e,0]
        b = G.edges[e,1]
        d = G.weights[e]/np.sqrt(LNorm[a]*RNorm[b])
        G.weights[e]=d
    
    return LNorm,RNorm


def LE(G,dim,verbose=0,maxiter=1000):
    """
    Laplacian Embedding of the data
    returns the dim-dimensional LE of the graph G
    chart = LE(G,dim,verbose=0,maxiter=1000)
    INPUT:
    - G : Weighted graph that represents the data
    - dim=1 : number of dimensions
    - verbose = 0: verbosity level
    - maxiter=1000: maximum number of iterations of the algorithm 
    OUTPUT
    - chart, array of shape(G.V,dim)
    NOTE :
    In fact the current implementation retruns
    what is now referred to a diffusion map at time t=1
    """
    n = G.V
    dim = np.minimum(dim,n)
    chart = nr.randn(G.V,dim+2)
    f = ff.Field(G.V,G.edges,G.weights,chart)
    LNorm,RNorm = local_sym_normalize(G)
    # nb : normally Rnorm = Lnorm
    if verbose:
        print np.sqrt(np.sum((LNorm-RNorm)**2))/np.sum(LNorm)
    eps = 1.e-7

    f1 = np.zeros((G.V,dim+2))
    for i in range(maxiter):
        f.diffusion(10)
        f0 = Orthonormalize(f.field)
        f.field = f0
        if nl.norm(f1-f0)<eps:
            break
        else:
            f1 = f0

    if verbose:
        print i,nl.norm(f1-f0)
    f.diffusion()
    f0 = f.field
    
    U,S,V = nl.svd(f0,0)
    RNorm = np.reshape(np.sqrt(RNorm),(n,1))
    chart = S[:dim]*np.repeat(1./RNorm,dim,1)*U[:,1:dim+1]
    chart = chart/np.sqrt(np.sum(chart**2,0))
            
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.bar(np.arange(np.size(S)-1),np.sqrt(1-S[1:]))
        mp.show()
        print "laplacian eigenvalues: ",1-S
        
    return chart

def LPP(G,X,dim,verbose=0,maxiter=1000):
    """
    Compute the Locality preserving projector of the data
    proj = LPP(G,X,dim,verbose=0,maxiter=1000)
    INPUT:
    - G : Weighted graph that represents the data
    - X : related input dataset
    - dim=1 : number of dimensions
    - verbose = 0: verbosity level
    - maxiter=1000: maximum number of iterations of the algorithm 
    OUTPUT
    -proj, array of shape(X.shape[1],dim)
    """
    n = G.V
    dim = np.minimum(dim,n) 
    G = fg.WeightedGraph(G.V,G.edges,G.weights)
    W = G.adjacency()
    D = np.diag(np.sum(W,1))
    M1 = np.dot(np.dot(np.transpose(X),D-W),X)
    M2 = np.dot(np.dot(np.transpose(X),D),X)
    C = nl.cholesky(M2)
    iC = nl.pinv(C)
    M1 = np.dot(iC,np.dot(M1,np.transpose(iC)))
    M2 = np.dot(iC,np.dot(M2,np.transpose(iC)))
    
    
    U,S,V = nl.svd(M1,0)
    if verbose:
        print S
    
    proj = np.dot(np.transpose(iC),U)
    proj = np.vstack([proj[:,-1-i] for i in range(dim)])
    proj = np.transpose(proj)
    proj = proj/np.sqrt(np.sum(proj**2,0))

    return proj


# ------------------------------------------------------------
# --------------- main classes -------------------------------
# ------------------------------------------------------------

class NLDR:
    """
    This is a generic class for dimension reduction techniques
    the main fields are
    - train_data : the input dataset from which the DR is perfomed
    - fdim=1
    - rdim=1
    """
    def __init__(self, X=None, rdim = 1, fdim=1):
        self.train_data = X
        if X!=None:
            self.fdim = X.shape[1] 
        else:
            self.fdim = fdim
        self.rdim = rdim
        self.trained = 0
        if self.fdim<self.rdim:
            raise ValueError, "reduced dim cannot be lower than fdim"
    
    def check_data(self,X):     
        if X.shape[1]!= self.fdim:
            raise ValueError, "Shape(X,1)=%d is not equal to fdim=%d"%(X.shape[1],self.fdim)
    
    def set_train_data(self,X):
        self.check_data(X)
        self.train_data = X
    
    def train(self):
        self.trained = 1
        
    def test(self,X):
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        self.check_data(X)
        
class MDS(NLDR):
    """
    This is a particular class that perfoms linear dimension reduction
    using multi-dimensional scaling
    besides the fields of NDLR, it contains the following ones:
    - trained: trained==1 means that the system has been trained 
    and can generalize
    - embedding: array of shape (nbitems,rdim)
    this is representation of the training data
    - offset: array of shape(nbitems)
    affine part of the embedding
    - projector: array of shape(fdim,rdim)
    linear part of the embedding
    """
    
    def train(self,verbose=0):
        """
        chart = MDS.train(verbose=0)
        verbose=0 : verbosity mode
        chart: resulting rdim-dimensional represntation
        """
        self.check_data(self.train_data)
        d = Euclidian_distance(self.train_data)
        u,v,rm = mds(d,self.rdim,verbose)
        self.trained = 1
        self.embedding = u
        self.offset = rm
        self.projector = v
        return(u)
    
    def test(self,X):
        """
        chart = MDS.test(X,verbose=0)
        X = array of shape(nbitems,fdim) 
        new data points to be embedded
        verbose=0 : verbosity mode
        chart: resulting rdim-dimensional represntation
        """
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        if np.size(X)==self.fdim:
            X = np.reshape(X,(1,self.fdim))
        self.check_data(X)
        d = Euclidian_distance(self.train_data,X)
        d = d*d
        d = d-np.reshape(self.offset,(self.train_data.shape[0],1))
        d = d-np.mean(d,0)
        u = np.dot(np.transpose(d),self.projector)
        return u[:,:self.rdim]
                    

class knn_Isomap(NLDR):
    """
    This is a particular class that perfoms linear dimension reduction
    using k nearest neighbor modelling and isomapping.
    besides the fields of NDLR, it contains the following ones:
    - k : number of neighbors in the knn graph building
    - G : resulting graph based on the training data
    - trained: trained==1 means that the system has been trained 
    and can generalize
    - embedding: array of shape (nbitems,rdim)
    this is representation of the training data
    - offset: array of shape(nbitems)
    affine part of the embedding
    - projector: array of shape(fdim,rdim)
    linear part of the embedding
    """
    
    def train(self,k=1,p=300,verbose=0):
        """
        chart = knn_Isomap.train(verbose=0)
        INPUT:
        - k=1 : k in the knn system
        - p=300 : number points used in the low dimensional approximation
        - verbose=0 : verbosity mode
        OUTPUT:
        - chart = knn_Isomap.embedding
        """
        self.k = k 
        self.check_data(self.train_data)
        
        #d = Euclidian_distance(self.train_data)
        n = self.train_data.shape[0]
        G = fg.WeightedGraph(n)
        G.knn(self.train_data,k)
        u,v,rm = isomap(G,self.rdim,p,verbose)

        self.G = G  
        self.trained = 1
        self.embedding = u
        self.offset = rm
        self.projector = v
        return(u)
    
    def test(self,X):
        """
        chart = knn_Isomap.test(X,verbose=0)
        INPUT
        X = array of shape(nbitems,fdim) 
        new data points to be embedded
        verbose=0 : verbosity mode
        OUTPUT
        chart: resulting rdim-dimensional represntation
        """     
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        if np.size(X)==self.fdim:
            X = np.reshape(X,(1,self.fdim))
        self.check_data(X)
        #
        n = self.G.V
        p = X.shape[0]
        G1 = self.G
        b,a,d = fg.graph_cross_knn(X,self.train_data,self.k)
        G1.V = n + p
        G1.E = G1.E + np.size(d)
        G1.edges = np.vstack((G1.edges,np.transpose(np.vstack((a,b+n)))))
        G1.edges = np.vstack((G1.edges,np.transpose(np.vstack((b+n,a)))))
        G1.weights = np.hstack((G1.weights,d))
        G1.weights = np.hstack((G1.weights,d))
        #
        d = np.zeros((p+n,p))
        for q in range(p):
            d[:,q] = G1.dijkstra(q+n)
        
        d = d[:n,:]
        d = d*d
        d = d-np.reshape(self.offset,(self.train_data.shape[0],1))
        d = d-np.mean(d,0)
        u = np.dot(np.transpose(d),self.projector)
        return u[:,:self.rdim]

class eps_Isomap(NLDR):
    """
    This is a particular class that perfoms linear dimension reduction
    using eps-ball neighbor modelling and isomapping.
    besides the fields of NDLR, it contains the following ones:
    - eps : eps-ball model used in the knn graph building
    - G : resulting graph based on the training data
    - trained: trained==1 means that the system has been trained 
    and can generalize
    - embedding: array of shape (nbitems,rdim)
    this is representation of the training data
    - offset: array of shape(nbitems)
    affine part of the embedding
    - projector: array of shape(fdim,rdim)
    linear part of the embedding
    """
    
    def train(self,eps=1.0,p=300,verbose=0):
        """
        chart = eps_Isomap.train(X,verbose=0)
        INPUT
        eps= 1.0: self.eps
        p = 300  number points used in the low dimensional approximation
        - verbose=0 : verbosity mode
        OUTPUT:
        - chart = eps_Isomap.embedding
        """     
        self.eps = eps
        self.check_data(self.train_data)
        
        #d = Euclidian_distance(self.train_data)
        n = self.train_data.shape[0]
        G = fg.WeightedGraph(n)
        G.eps(self.train_data,self.eps)
        u,v,rm = isomap(G,self.rdim,p,verbose)

        self.G = G  
        self.trained = 1
        self.embedding = u
        self.offset = rm
        self.projector = v
        return(u)
    
    def test(self,X):
        """
        chart = eps_Isomap.test(X,verbose=0)
        INPUT
        X = array of shape(nbitems,fdim) 
        new data points to be embedded
        verbose=0 : verbosity mode
        OUTPUT
        chart: resulting rdim-dimensional represntation
        """
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        if np.size(X)==self.fdim:
            X = np.reshape(X,(1,self.fdim))
        self.check_data(X)
        #
        n = self.G.V
        p = X.shape[0]
        G1 = self.G
        b,a,d = fg.graph_cross_eps(X,self.train_data,self.eps)
        G1.V = n + p
        G1.E = G1.E + np.size(d)
        G1.edges = np.vstack((G1.edges,np.transpose(np.vstack((a,b+n)))))
        G1.edges = np.vstack((G1.edges,np.transpose(np.vstack((b+n,a)))))
        G1.weights = np.hstack((G1.weights,d))
        G1.weights = np.hstack((G1.weights,d))
        #
        d = np.zeros((p+n,p))
        for q in range(p):
            d[:,q] = G1.dijkstra(q+n)
        
        d = d[:n,:]
        d = d*d
        d = d-np.reshape(self.offset,(self.train_data.shape[0],1))
        d = d-np.mean(d,0)
        u = np.dot(np.transpose(d),self.projector)
        return u[:,:self.rdim]


class knn_LE(NLDR):
    """
    This is a particular class that perfoms linear dimension reduction
    using k nearest neighbor modelling and laplacian embedding.
    besides the fields of NDLR, it contains the following ones:
    - k : number of neighbors in the knn graph building
    - G : resulting graph based on the training data
    - trained: trained==1 means that the system has been trained 
    and can generalize
    - embedding: array of shape (nbitems,rdim)
    this is representation of the training data
    NB: to date, only the training part (embedding computation) is considered
    """
    def train(self,k=1,verbose=0):
        """
        chart = knn_LE.train(k=1,verbose=0)
        INPUT:
        - k=1 : k in the knn system
        - verbose=0 : verbosity mode
        OUTPUT:
        - chart = knn_LE.embedding
        """
        self.k = k 
        self.check_data(self.train_data)
        
        n = self.train_data.shape[0]
        G = fg.WeightedGraph(n)
        G.knn(self.train_data,k)
        G.set_gaussian(self.train_data)
        u = LE(G,self.rdim,verbose=verbose)

        self.G = G  
        self.trained = 1
        self.embedding = u
        return(u)
    
class knn_LPP(NLDR):
    """
    This is a particular class that perfoms linear dimension reduction
    using k nearest neighbor modelling and locality preserving projection (LPP).
    besides the fields of NDLR, it contains the following ones:
    - k : number of neighbors in the knn graph building
    - G : resulting graph based on the training data
    - trained: trained==1 means that the system has been trained 
    and can generalize
    - embedding: array of shape (nbitems,rdim)
    this is representation of the training data
    - projector: array of shape(fdim,rdim)
    linear part of the embedding
    """
    def train(self,k=1,verbose=0):
        """
        chart = knn_LPP.train(verbose=0)
        INPUT:
        - k=1 : k in the knn system
        - verbose=0 : verbosity mode
        OUTPUT:
        - chart = knn_LPP.embedding
        """
        self.k = k 
        self.check_data(self.train_data)
        
        n = self.train_data.shape[0]
        G = fg.WeightedGraph(n)
        G.knn(self.train_data,k)
        G.set_gaussian(self.train_data)
        self.projector = LPP(G,self.train_data,self.rdim)

        self.embedding = np.dot(self.train_data,self.projector)
        self.trained = 1
        return(self.embedding)

    def test(self,X):
        """
        chart = knn_LPP.test(X,verbose=0)
        INPUT
        X = array of shape(nbitems,fdim) 
        new data points to be embedded
        verbose=0 : verbosity mode
        OUTPUT
        chart: resulting rdim-dimensional represntation
        """     
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        if np.size(X)==self.fdim:
            X = np.reshape(X,(1,self.fdim))
        self.check_data(X)
        #
        u = np.dot(X,self.projector)
        return u

#------------------------------------------------------------------
#------- Ancillary methods ----------------------------------------
#------------------------------------------------------------------

def check_isometry(G,chart,nseeds=100,verbose = 0):
    """
    A simple check of the Isometry:
    look whether the output distance match the intput distances
    for nseeds points
    OUTPUT:
    - a proportion factor to optimize the metric
    """
    nseeds = np.minimum(nseeds, G.V)
    aux = np.argsort(nr.rand(nseeds))
    seeds =  aux[:nseeds]
    dY = Euclidian_distance(chart[seeds],chart)
    dX = G.floyd(seeds)

    dY = np.reshape(dY,np.size(dY))
    dX = np.reshape(dX,np.size(dX))

    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(dX,dY,'.')
        mp.show()

    scale = np.dot(dX,dY)/np.dot(dX,dX)
    return scale
    

def local_correction_for_embedding(G,chart,sigma = 1.0):
    """
    WIP : an unfinished fuction that aims at improving isomap's prbs
    the idea is to optimize the representation of local distances
    INPUT:
    G: the graph to be isomapped
    chart: the input chart
    sigma: a scale parameter
    OUTPUT:
    chart : the corrected chart
    """
    sqsigma = 2*sigma**2
    seeds = np.arange(G.V)
    dX = G.floyd(seeds)
    weight = np.exp(-dX**2/sqsigma)
    dY = Euclidian_distance(chart[seeds],chart)
    criterion = np.sum(weight*(dX-dY)**2)/G.V
    print criterion
    tiny = 1.e-10

    for i in range(30):
        aux = weight*(dX-dY)/np.maximum(dY,tiny)
        grad = np.zeros(np.shape(chart))
        for j in range(G.V):
            grad[j,:] = np.dot(aux[j,:],chart-chart[j,:])

        chart = chart - grad/G.V
        dY = Euclidian_distance(chart[seeds],chart)
        criterion = np.sum(weight*(dX-dY)**2)/G.V             
        print np.sum(grad**2)/G.V,criterion
    return chart

def sparse_local_correction_for_embedding(G,chart,sigma = 1.0,niter=100):
    """
    WIP : an unfinished fuction that aims at improving isomap's prbs
    the idea is to optimize the representation of local distances
    INPUT:
    G: the graph to be isomapped
    chart: the input chart
    sigma: a scale parameter
    OUTPUT:
    chart : the corrected chart
    NOTE:
    the graph G is reordeered
    """
    G.reorder(0)
    sqsigma = 2*sigma**2
    dX = G.weights
    weight = np.exp(-dX**2/sqsigma)

    K = G.copy()
    K.set_euclidian(chart)
    dY = K.weights
    dXY= dX-dY
    criterion = np.sum(weight*(dX-dY)**2)/G.V

    tiny = 1.e-10
    ln = G.list_of_neighbors()
    ci, ne, we = G.to_neighb()
    
    for i in range(niter):
        aux = weight*(dX-dY)/np.maximum(dY,tiny)        
        grad = np.zeros(np.shape(chart))
        for j in range(G.V):
            k = ln[j]
            grad[j,:] = np.dot(aux[ci[j]:ci[j+1]],np.squeeze(chart[k,:]-chart[j,:]))/np.size(k)

        ngrad = np.sum(grad**2,1)
        chart = chart - grad
        K.set_euclidian(chart)
        dY = K.weights
        criterion = np.sum(weight*(dX-dY)**2)/G.V
    return chart
    
def partial_floyd_graph(G,k):
    """
    Create a graph of the knn in teh geodesic sense, given an input graph G
    """
    ls = []
    ln = []
    ld = []
    for i in range(G.V):
        dg = G.dijkstra(i)      
        if k<G.V:
            sdg = np.sort(dg)
            tdg = sdg[k]
        else:
            tdg = dg.max()+1
        j = np.flatnonzero(dg<tdg)
        ln.append(j)
        ld.append(dg[j])
        ls.append(i*np.ones(len(j)))

    ln = np.concatenate(ln)
    ld = np.concatenate(ld)
    ls = np.concatenate(ls)
    edges = np.transpose(np.vstack((ls,ln)))
    K = fg.WeightedGraph(G.V,edges,ld)
    K.symmeterize()
    return K


# --------------------------------------------------------------------
# ---------- test part ---------------------------------------------
# --------------------------------------------------------------------

def _swiss_roll(nbitem=1000):
    """
    Sample nbitem=1000 point from a swiss roll
    Ouput
    - X (nbsamp,3) array giving the 3D embedding
    - x (nbsamp,2) array giving the intrinsic coordinates
    """
    x = nr.rand(nbitem,2)
    X1 = np.reshape((1+x[:,0])*(np.cos(2*np.pi*x[:,0])),(nbitem,1))
    X2 = np.reshape((1+x[:,0])*(np.sin(2*np.pi*x[:,0])),(nbitem,1))
    X3 = np.reshape(10*x[:,1],(nbitem,1))
    X = np.hstack((X1,X2,X3))
    return X,x

def _orange(nbsamp=1000):
    """
    Sample nbsamp=1000 points from an 'orange' (a sphere with two partial cuts)
    output : the corresponding graph
    """
    # make the sphere
    x = nr.randn(nbsamp,3)
    X = np.transpose(x)/np.sqrt(np.sum(x**2,1))
    X = np.transpose(X)
    G = fg.WeightedGraph(nbsamp)
    G.knn(X,16)

    #make the cuts
    OK = np.ones(G.E)
    for e in range(G.E):
        X1 = X[G.edges[e,0]]
        X2 = X[G.edges[e,1]]
        #only in the positive z>0 half space
        if ((X1[2]>0) & (X2[2]>0)):
            if  X1[1]*X2[1]<0:
                OK[e]=0
            if  X1[0]*X2[0]<0:
                OK[e]=0

    G.remove_edges(OK)
    return G,X
    
def _test_isomap_orange(verbose=0):
    """
    A test of the isomap new look using a swiss roll
    """
    nbsamp = 1000
    G,X = _orange(nbsamp)
    nbseed = 1000#nbsamp
    rdim = 3
    u = isomap_dev(G,rdim,nbseed,1)
    check_isometry(G,u[:,:2],nseeds  =100)
    #v = local_correction_for_embdedding(G,u[:,:2],sigma = 1.0)
    K = partial_floyd_graph(G,300)
    v = sparse_local_correction_for_embedding(K,u[:,:2],sigma = 1.0) 
    check_isometry(G,v,nseeds  =100)
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(u[:,0],u[:,1],'.')
        mp.plot(v[:,0],v[:,1],'.r')
        mp.show()

    
def _test_cca():
    """
    Basic (valid) test of the CCA
    """
    X = nr.randn(100,3)
    Y = nr.randn(100,3)
    cc1 = CCA(X,Y)
    A = nr.randn(3,3)
    Z = np.dot(X,A)
    cc2 = CCA(X,Z)
    cc3 = CCA(Y,Z)
    test = (np.sum((cc1-cc3)**2)<1.e-7)&(np.min(cc2>1.e-7))
    return test



def _test_isomap_dev():
    """
    A test of the isomap new look using a swiss roll
    """
    nbsamp = 1000
    X,x = _swiss_roll(nbsamp)
    G = fg.WeightedGraph(nbsamp)
    G.knn(X,8)
    nbseed = 300
    rdim = 3
    u = isomap_dev(G,rdim,nbseed,1)
    sv = CCA(x,u[:,:3])
    check_isometry(G,u[:,:2],nseeds  =100)
    return (sv.sum()>1.9)

def _test_mds():
    X = nr.randn(10,3)
    M = MDS(X,rdim=2)
    u = M.train()
    x = X[:2,:]
    a = M.test(x)
    eps = 1.e-12
    test = np.sum(a-u[:2,:])**2<eps
    return test

def _test_knn_isomap():
    X = nr.randn(10,3)
    M = knn_Isomap(X,rdim=1)
    u = M.train(k=2)
    x = X[:2,:]
    a = M.test(x)
    eps = 1.e-12
    test = np.sum(a-u[:2,:])**2<eps
    return test
    
def _test_eps_isomap():
    """
    Test of the esp_isompa procedure
    To be checked: returns FALSE
    """
    X = nr.randn(10,3)
    M = eps_Isomap(X,rdim=1)
    u = M.train(eps = 2.)
    x = X[:2,:]
    a = M.test(x)
    eps = 1.e-12
    test = np.sum(a-u[:2,:])**2<eps
    if test==False:
        print np.sum(a-u[:2,:])**2
    return test
    
def _test_knn_LE_():
    """
    Test of  eth Laplacian embedding 
    """
    X = nr.randn(100,3)
    M = knn_LE(X,rdim=1)
    u = M.train(k=5)

    
def _test_knn_LE(verbose=0):
    """
    a swiss roll example to validate the method
    """
    nbitem = 1000
    x = nr.rand(nbitem,3)
    X1 = np.reshape((1+x[:,0])*(np.cos(2*np.pi*x[:,0])),(nbitem,1))
    X2 = np.reshape((1+x[:,0])*(np.sin(2*np.pi*x[:,0])),(nbitem,1))
    X3 = np.reshape(10*x[:,1],(nbitem,1))
    X = np.hstack((X1,X2,X3))
    M = knn_LE(X,rdim=3)
    u = M.train(k=8,verbose=verbose)

    sv = CCA(x,u)
    if verbose:
        print "canonical correlations: ", sv
        import pylab as p
        import matplotlib.axes3d as p3
        fig=p.figure()
        ax = p3.Axes3D(fig)
        #datap = xyz[:,idx].astype('float')
        ax.scatter3D(np.squeeze(X1),np.squeeze(X2),np.squeeze(X3))
        p.show()
    
    
def _test_knn_LPP():
    X = nr.randn(10,3)
    M = knn_LPP(X,rdim=1)
    u = M.train(k=5)
    x = X[:2,:]
    a = M.test(x)
    eps = 1.e-12
    test = np.sum(a-u[:2,:])**2<eps
    return test
    
# --------------------------------------------------------------------
# ------------- WIP --------------------------------------------------
# --------------------------------------------------------------------



def _test_lE0(verbose=0):
    n = 100
    dim = 2
    k= 5
    X = nr.randn(n,dim)
    G = fg.WeightedGraph(n)
    G.knn(X,5)
    G.set_gaussian(X)
    LE(G,5,verbose)

def _test_lE1(verbose=0):
    n = 100
    t = 2*np.pi*nr.randn(n)
    X = np.transpose(np.vstack((np.cos(t),np.sin(t),t)))
    X = X+0.1*nr.randn(X.shape[0],X.shape[1])
    G = fg.WeightedGraph(n)
    G.knn(X,10)
    G.set_gaussian(X)
    u = LE(G,dim=5)
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(t,u[:,0],'.')
        mp.figure()
        mp.plot(u[:,0],u[:,1],'.')
        mp.figure()
        mp.plot(X[:,0],X[:,1],'.')
    x = G.cc();
    print x.max()

def _test_LPP(verbose=0):
    n = 100
    t = 2*np.pi*nr.randn(n)
    X = np.transpose(np.vstack((np.cos(t),np.sin(t),t)))
    X = X+0.1*nr.randn(X.shape[0],X.shape[1])
    G = fg.WeightedGraph(n)
    G.knn(X,3)
    G.set_gaussian(X)
    u = LPP(G,X,dim=2)
    em = np.dot(X,u)
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(em[:,0],em[:,1],'.')

def _test_LDC(verbose=0):
    """
    Generate a dataset with latent dimension=3
    and plot
    """
    n = 200
    dim = 10
    r = 3
    X = nr.randn(n,dim)#
    for k in range(r):
        X[:,k] += 2*nr.randn(n)

    infer_latent_dim(X,verbose) 

