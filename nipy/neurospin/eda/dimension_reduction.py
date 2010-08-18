# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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

Bertrand Thirion, 2006-2009
"""

import numpy as np
import scipy.linalg as nl
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
    
    Parameters
    ----------
    l array of shape (n) spectrum
    k, int,  test rank (?)
    dimf, int, maximal rank (?)
    n, int, number of inputs (?)
    
    Returns
    -------
    ll, float, The log-likelihood
    
    Note
    ---- 
    This is imlpempented from Minka et al., 2001
    """
    if k>dimf:
        raise ValueError, "the dimension cannot exceed dimf"
    from scipy.special import gammaln
    Pu = -k*np.log(2)
    for i in range(k):
        Pu += gammaln((dimf-i)/2)-np.log(np.pi)*(dimf-i)/2
        
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
    l_ = l.copy()
    l_[k:dimf] = v
    for i in range(k):
        for j in range (i+1,dimf):
            Pa = Pa + np.log((l[i]-l[j])*(1./l_[j]-1./l_[i]))+np.log(n)

    Pa = -Pa/2
    lE = Pu+Pl+Pv+Pp+Pa-k*np.log(n)/2

    return lE

def infer_latent_dim(X, verbose=0, maxr=-1):
    """
    r = infer_latent_dim(X, verbose=0)
    Infer the latent dimension of an aray assuming data+gaussian noise mixture
    
    Parameters
    ----------
    array X, data whose deimsnionhas to be inferred
    verbose=0, int, verbosity level
    maxr=-1, int, maximum dimension that can be achieved
             if maxr = -1, this is equal to rank(X)
    
    Returns
    -------
    r, int, the inferred dimension
    """
    if maxr ==-1:
        maxr = np.minimum(X.shape[0],X.shape[1])
        
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

    return rank


def Euclidian_distance(X, Y=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vector
    
    Parameters
    ----------
    X, array of shape (n1,p)
    Y=None, array of shape (n2,p)
            if Y==None, then Y=X is used instead

    Returns
    -------
    ED, array fo shape(n1, n2)
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

def CCA(X, Y, eps=1.e-15):
    """
    Canonical corelation analysis of two matrices
    
    Parameters
    ----------
    X array of shape (nbitem,p) 
    Y array of shape (nbitem,q) 
    eps=1.e-15, float is a small biasing constant
                to grant invertibility of the matrices
    
    Returns
    -------
    ccs, array of shape(min(n,p,q) the canonical correlations
        
    Note
    ----
    It is expected that nbitem>>max(p,q)
    """
    from numpy.linalg import cholesky, inv, svd
    if Y.shape[0]!=X.shape[0]:
        raise ValueError,"Incompatible dimensions for X and Y"
    p = X.shape[1]
    q = Y.shape[1]
    sqX = np.dot(X.T,X)
    sqY = np.dot(Y.T,Y)
    sqX += np.trace(sqX)*eps*np.eye(p)
    sqY += np.trace(sqY)*eps*np.eye(q)
    rsqX = cholesky(sqX)
    rsqY = cholesky(sqY)
    iX = inv(rsqX).T
    iY = inv(rsqY).T
    Cxy = np.dot(np.dot(X,iX).T,np.dot(Y,iY))
    uv, ccs, vv = svd(Cxy)
    return ccs


# ------------------------------------------------------------------------
# --- Multi_simensional scaling ------------------------------------------
# ------------------------------------------------------------------------

def Euclidian_mds(X, dim, verbose=0):
    """
    returns a dim-dimensional MDS representation of the rows of X 
    using an Euclidian metric 
    """
    d = Euclidian_distance(X)
    return(mds(d, dim, verbose))
    
def mds_parameters(dm, dim=1, verbose=0):
    """
    Compute the embedding parameters to perform the embedding of some data
    given a distance matrix

    Parameters
    ----------
    dm, array of shape(nbitem, nbitem), the input distance matrix
    dim=1: the dimension of the desired representation
    verbose=0: verbosity level

    Returns
    -------
    embedding_direction, array of shape (nbitem, dim) 
                         the set of directions used to embed the data
                         in the reduced space        
    scaling, array of shape (dim)
             scaling to apply to perform the embedding of the data
    offset,  array of shape (nbitem)
             additive factor necessary to center the embedding
             (analogous to mean subtraction)

    """
    # take the squared distance and center the matrix
    sqdm = 0.5*dm*dm
    msd = sqdm.mean(0)
    sqdm -= msd
    rm1 = sqdm.mean(1)
    sqdm = (sqdm.T-rm1).T

    U,S,V = nl.svd(sqdm,0)
    sqs = np.sqrt(S)
    embedding_direction = V.T [:,:dim]
    scaling = sqs[:dim]
    offset  = msd
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.bar(np.arange(np.size(sqs)),sqs)

    return embedding_direction, scaling, offset

def mds_embedding(dm, embedding_direction, scaling, offset):
    """
    Embedding new data  based on a previous mds computation

    Parameters
    ----------
    dm, array of shape(nb_test_item, nb_train_item), 
        the input distance matrix: 
    embedding_direction, array of shape (nb_train_item, dim) 
                         the set of directions used to embed the data
                         in the reduced space        
    scaling, array of shape (dim)
             scaling to apply to perform the embedding of the data
    offset,  array of shape (nbitem)
             additive factor necessary to center the embedding
             (analogous to mean subtraction)
    
    Returns
    -------
    chart, array of shape(nb_test_item, dim),
           the resulting reprsentation of the traing data
    
    """
    sqdm = 0.5*dm*dm - offset
    rm1 = sqdm.mean(1)
    sqdm = (sqdm.T-rm1).T    
    chart = np.dot(sqdm, embedding_direction)/scaling
    return chart

def mds(dm, dim=1, verbose=0):
    """
    Multi-dimensional scaling, i.e. derivation of low dimensional
    representations from distance matrices.
    
    Parameters
    ----------
    dm, array of shape(nbitem, nbitem), the input distance matrix
    dim=1: the dimension of the desired representation
    verbose=0: verbosity level

    Returns
    -------
    chart, array of shape(nbitem, dim),
           the resulting reprsentation of the traing data
    embedding_direction, array of shape (nbitem, dim) 
                         the set of directions used to embed the data
                         in the reduced space        
    scaling, array of shape (dim)
             scaling to apply to perform the embedding of the data
    offset,  array of shape (nbitem)
             additive factor necessary to center the embedding
             (analogous to mean subtraction)
    """
    
    # get the parameters
    embedding_direction, scaling, offset = mds_parameters(dm, dim, verbose)

    # perform the embedding of new data
    chart = mds_embedding(dm, embedding_direction, scaling, offset)

    return chart, embedding_direction, scaling, offset

def isomap(G, dim=1, p=300, verbose=0):
    """
    Isomapping of the data
    return the dim-dimensional ISOMAP chart that best represents the graph G
    
    Parameters
    ----------
    G : nipy.neurospin.graph.WeightedGraph instance that represents the data
    dim=1, int number of dimensions
    p=300, int nystrom reduction of the problem
    verbose = 0: verbosity level
    
    Returns
    -------
    chart, array of shape(nbitem, dim),     
           the resulting reprsentation of the traing data
    proj, array of shape (nbitem, dim) 
          the set of directions used to embed the data
          in the reduced space        
    scaling, array of shape (dim)
             scaling to apply to perform the embedding of the data
    offset,  array of shape (nbitem)
             additive factor necessary to center the embedding
             (analogous to mean subtraction)
    seed, array of shape (p): 
          the seed nodes that were used to compute fast embedding
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
        seed = np.arange(n)
    
    #print G.edges, G.weights, G.E
    chart, proj, scaling, offset = mds(dg.T,dim,verbose)
    
    return chart, proj, scaling, offset, seed



def LE_dev(G, dim, verbose=0, maxiter=1000):
    """
    Laplacian Embedding of the data
    returns the dim-dimensional LE of the graph G
    
    Parameters
    ----------
    G, nipy.neurospin.graph.WeightedGraph instance that represents the data
    dim=1, int number of dimensions
    verbose=0, verbosity level
    maxiter=1000, maximum number of iterations of the algorithm 
    
    Returns
    -------
    chart, array of shape(G.V,dim), the resulting embedding
    """
    n = G.V
    dim = np.minimum(dim,n)
    chart = nr.randn(G.V,dim+2)
    f = ff.Field(G.V,G.edges,G.weights,chart)
    S = f.normalize(0)
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

    LE = np.sqrt(np.sum(f0**2,0))
    U = Orthonormalize(f.field)
    chart = np.transpose(np.transpose(U[:,1:dim+1])/S)
    chart = chart/np.sqrt(np.sum(chart**2,0))


    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.bar(np.arange(np.size(LE)-1),np.sqrt(1-LE[1:]))
        print 1-LE
        
    return chart


def Orthonormalize(M):
    """
    M = Orthonormalize(M)
    orthonormalize the columns of M
    (Gram-Schmidt procedure)
    """
    C = nl.cholesky(np.dot(M.T,M))
    M = np.dot(M,(nl.inv(C)).T)
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


def LE(G, dim, verbose=0, maxiter=1000):
    """
    Laplacian Embedding of the data
    returns the dim-dimensional LE of the graph G
    
    Parameters
    ----------
    G, nipy.neurospin.graph.WeightedGraph instance that represents the data
    dim=1, int, number of dimensions
    verbose=0, verbosity level
    maxiter=1000, int, maximum number of iterations of the algorithm 
    
    Returns
    -------
    chart, array of shape(G.V,dim)
    
    Note
    ----
    In fact the current implementation retruns
    what is now referred to a diffusion map at time t=1
    """
    n = G.V
    dim = np.minimum(dim,n)
    chart = nr.randn(G.V,dim+2)
    f = ff.Field(G.V,G.edges,G.weights,chart)
    LNorm,RNorm = local_sym_normalize(G)
    # note : normally Rnorm = Lnorm
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
        print "laplacian eigenvalues: ",1-S
        
    return chart

def LPP(G, X, dim, verbose=0, maxiter=1000):
    """
    Compute the Locality preserving projector of the data
    
    Parameters
    ----------
    G, nipy.neurospin.graph.WeightedGraph instance that represents the data
    X, array of shape (G.V, input_dim) related input dataset
    dim=1, int, representation dimensions
    verbose=0, bool verbosity level
    maxiter=1000, int, maximum number of iterations of the algorithm 
    
    Returns
    -------
    proj, array of shape(X.shape[0],dim)

    Note
    ----
    non-sparse version ; breaks for large number of input items
    """
    n = G.V
    dim = np.minimum(dim,n) 
    G = fg.WeightedGraph(G.V,G.edges,G.weights)
    W = G.adjacency()
    D = np.diag(np.sum(W,1))
    M1 = np.dot(np.dot(X.T,D-W),X)
    M2 = np.dot(np.dot(X.T,D),X)
    C = nl.cholesky(M2)
    iC = nl.pinv(C)
    M1 = np.dot(iC,np.dot(M1,iC.T))
    M2 = np.dot(iC,np.dot(M2,iC.T))
    
    
    U,S,V = nl.svd(M1,0)
    if verbose:
        print S
    
    proj = np.dot(iC.T,U)
    proj = np.vstack([proj[:,-1-i] for i in range(dim)])
    proj = proj.T
    proj = proj/np.sqrt(np.sum(proj**2,0))

    return proj


# ------------------------------------------------------------
# --------------- main classes -------------------------------
# ------------------------------------------------------------

class NLDR:
    """
    This is a generic class for non-linear dimension reduction techniques
         (NLDR) the main members are:
    train_data, array the input dataset from which the DR is perfomed
    fdim=1, int, the input deature dimension
    rdim=1, int, the reduced feature dimension
    trained: trained==1 means that the system has been trained 
             and can generalize
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
        """
        Check that X has the specified fdim
        """     
        if X.shape[1]!= self.fdim:
            raise ValueError, "Shape(X,1)=%d is not equal to fdim=%d" \
                  %(X.shape[1],self.fdim)
    
    def set_train_data(self,X):
        """
        Set the input array X as  the training data of the class
        """
        self.check_data(X)
        self.train_data = X
    
    def train(self):
        """
        set self.trained as 1
        """
        self.trained = 1
        
    def test(self,X):
        """
        check that X is suitable as test data
        """
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        self.check_data(X)

class PCA(NLDR):
    """
    This class performs PCA-based linear dimension reduction
    besides the fields of NDLR, it contains the following ones:
    
    embedding: array of shape (nbitems,rdim)
               this is representation of the training data
    offset: array of shape(nbitems)
            affine part of the embedding
    projector: array of shape(fdim,rdim)
               linear part of the embedding
    
    """

    def train(self, verbose=0):
        """
        training procedure
        
        Parameters
        ----------
        verbose=0 : verbosity mode
        
        Returns
        -------
        chart: resulting rdim-dimensional representation
        """
        from numpy.linalg import svd
        self.check_data(self.train_data)
        self.offset = self.train_data.mean(0)
        x = self.train_data-self.offset
        u,s,v  = svd(x,0)
        self.embedding = (u*s)[:,:self.rdim]
        self.scaling = s[:self.rdim]
        self.projector = v[:self.rdim,:].T
        self.trained = 1
        return(self.embedding)

    def test(self, x):
        """
        Apply the learnt embedding to the new data x
        
        Parameters
        ----------
        x: array of shape(nbitems, fdim) 
           data points to be embedded
        
        Returns
        -------
        chart: array of shape (nbitems, rdim) 
        resulting rdim-dimensional represntation
        """
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        if np.size(x)==self.fdim:
            x = np.reshape(x,(1,self.fdim))
        self.check_data(x)

        u = np.dot(x-self.offset,self.projector)
        
        return u            



class MDS(NLDR):
    """
    This is a particular class that perfoms linear dimension reduction
         using multi-dimensional scaling
         (PCA of the distance matrix)
    besides the fields of NDLR, it contains the following ones:
    
    embedding: array of shape (nbitems,rdim)
               this is representation of the training data
    offset: array of shape(nbitems)
            affine part of the embedding
    projector: array of shape(fdim,rdim)
               linear part of the embedding
    """
    
    def train(self, verbose=0):
        """
        training procedure
        
        Parameters
        ----------
        verbose=0 : verbosity mode
        
        Returns
        -------
        chart: resulting rdim-dimensional representation
        """
        self.check_data(self.train_data)
        dm = Euclidian_distance(self.train_data)
        u, v, sc, msd = mds(dm, self.rdim, verbose)
        self.trained = 1
        self.mean = np.reshape(self.train_data.mean(0),(1,self.fdim))
        self.embedding = u
        self.scaling = sc
        self.projector = v
        self.offset = msd
        return(u)
    
    def test(self, x):
        """
        Apply the learnt embedding to the new data x
        
        Parameters
        ----------
        x: array of shape(nbitems, fdim) 
           data points to be embedded
        
        Returns
        -------
        chart: array of shape (nbitems, rdim) 
        resulting rdim-dimensional represntation
        """
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        if np.size(x)==self.fdim:
            x = np.reshape(x,(1,self.fdim))
        self.check_data(x)
        
        # trivial solution -- valid only in the Euclidian case
        #dp = -np.dot(x-self.mean,(self.train_data-self.mean).T)
        #u = np.dot(dp,self.projector)/self.scaling
        
        dm = Euclidian_distance(x,self.train_data)
        u = mds_embedding(dm, self.projector, self.scaling, self.offset)

        return u            

class knn_Isomap(NLDR):
    """
    This is a particular class that perfoms linear dimension reduction
         using k-nearest-neighbor (knn) modelling and isomapping.
    Besides the fields of NDLR, it contains the following ones:
    
    k : number of neighbors in the knn graph building
    G : knn graph based on the training data
    embedding: array of shape (nbitems,rdim)
               this is representation of the training data
    offset: array of shape(nbitems)
            affine part of the embedding
    projector: array of shape(fdim,rdim)
               linear part of the embedding
    """
    
    def train(self, k=1, p=300, verbose=0):
        """
        Training function
        
        Parameters
        ----------
        k=1, int, k in the knn system
        p=300, int, number points used in the low dimensional approximation
        verbose=0, bool, verbosity mode
        
        Returns
        -------
        chart, array of shape (nbLearningSamples, rdim) 
               knn_Isomap embedding
        """
        self.k = k 
        self.check_data(self.train_data)
        
        n = self.train_data.shape[0]
        G = fg.WeightedGraph(n)
        G.knn(self.train_data, k)

        u, proj, scaling, offset, seed = isomap(G, self.rdim, p, verbose)

        self.seed = seed
        self.G = G  
        self.trained = 1
        self.embedding = u
        self.offset = offset
        self.projector = proj
        self.scaling = scaling
        return(u)
    
    def test(self,x):
        """
        embed new data into the learnt representation
        
        Parameters
        ----------
        x array of shape(nbitems,fdim) 
          new data points to be embedded
        verbose=0, bool, verbosity mode
        
        Returns
        -------
        chart, array of shape (nbitems, rdim) 
               resulting rdim-dimensional represntation
        """     
        # preliminary checks
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        if np.size(x)==self.fdim:
            x = np.reshape(x,(1,self.fdim))
        self.check_data(x)
        
        # launch the algorithm:
        # step 1: create a compound graph with the learning and test vertices
        n = self.G.V
        m = x.shape[0]
        G1 = self.G
        b, a, d = fg.graph_cross_knn(x,self.train_data,self.k)
        G1.V = n + m
        G1.E = G1.E + 2*np.size(d)
        G1.edges = np.vstack((G1.edges,np.transpose(np.vstack((a,b+n)))))
        G1.edges = np.vstack((G1.edges,np.transpose(np.vstack((b+n,a)))))
        G1.weights = np.hstack((G1.weights,d))
        G1.weights = np.hstack((G1.weights,d))
        
        # perform dijkstra's distance computation
        dg = np.zeros((m,n+m))
        for q in range(m):
            dg[q] = G1.dijkstra(q+n)
        dg = dg[:,self.seed]

        # perform the embedding based on these distances
        u = mds_embedding(dg, self.projector, self.scaling, self.offset)
        
        return u

class eps_Isomap(NLDR):
    """
    This is a particular class that perfoms linear dimension reduction
         using eps-ball neighbor modelling and isomapping.
    besides the fields of NDLR, it contains the following ones:
    
    eps, float eps-ball model used in the knn graph building
    G : data-representing graph learnt from the training data
    embedding: array of shape (nbitems,rdim)
               this is representation of the training data
    offset: array of shape(nbitems)
            affine part of the embedding
    projector: array of shape(fdim,rdim)
               linear part of the embedding
    """
    
    def train(self, eps=1.0, p=300, verbose=0):
        """
        Traing/learning function
        
        Parameters
        ----------
        eps=1.0, float value of self.eps
        p=300, int  
               number points used to compute the low dimensional approximation
        verbose=0 : verbosity mode
        
        returns
        -------
        chart, array of shape (nbLearningSamples, rdim),
               the resulting embedding
        """     
        self.eps = eps
        self.check_data(self.train_data)
        
        n = self.train_data.shape[0]
        G = fg.WeightedGraph(n)
        G.eps(self.train_data,self.eps)
        
        u, proj, scaling, offset, seed = isomap(G, self.rdim, p, verbose)

        self.seed = seed
        self.G = G  
        self.trained = 1
        self.embedding = u
        self.offset = offset
        self.projector = proj
        self.scaling = scaling
        return(u)
    
    def test(self,x):
        """
        Embedding the data conatined in x
        
        Parameters
        ----------
        x:array of shape(nbitems,fdim) 
                new data points to be embedded
        verbose=0, bool, verbosity mode
        
        Returns
        -------
        chart: array of shape (nbitems, rdim) 
               resulting rdim-dimensional represntation
        """
        # preliminary checks
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        if np.size(x)==self.fdim:
            x = np.reshape(x,(1,self.fdim))
        self.check_data(x)
        
        # construction of a graph with the training and test data
        n = self.G.V
        m = x.shape[0]
        G1 = self.G
        b, a, d = fg.graph_cross_eps(x, self.train_data, self.eps)
        G1.V = n + m
        G1.E += 2*np.size(d)
        G1.edges = np.vstack((G1.edges,np.transpose(np.vstack((a,b+n)))))
        G1.edges = np.vstack((G1.edges,np.transpose(np.vstack((b+n,a)))))
        G1.weights = np.hstack((G1.weights,d))
        G1.weights = np.hstack((G1.weights,d))
        
        # perform dijkstra's distance computation
        dg = np.zeros((m, n+m))
        for q in range(m):
            dg[q] = G1.dijkstra(q+n)
        dg = dg[:,self.seed]
       
        # perform the embedding based on these distances
        u = mds_embedding(dg, self.projector, self.scaling, self.offset)
        
        return u        

class knn_LE(NLDR):
    """
    This is a particular class that perfoms linear dimension reduction
         using k nearest neighbor modelling and laplacian embedding.
    besides the fields of NDLR, it contains the following ones:
    
    k, int number of neighbors in the knn graph building
    G, graph lerant on the training data
    embedding: array of shape (nbitems,rdim)
               this is representation of the training data
    fixme: to date, only the training part (embedding computation) 
           is considered
    """
    def train(self, k=1, verbose=0):
        """
        learning function
        
        Parameters
        ----------
        k=1, int k in the knn system
        verbose=0, bool, verbosity mode
        
        Returns
        -------
        chart: array of shape (nblearningSamples, rdim) 
               embedding of the training data
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
        using k nearest neighbor modelling and locality preserving projection 
        (LPP).
    besides the fields of NDLR, it contains the following ones:
    
    k, int, number of neighbors in the knn graph building
    G, graph learnt based on the training data
    embedding: array of shape (nbitems,rdim)
               this is representation of the training data
    projector: array of shape(fdim,rdim)
               linear part of the embedding
    """
    def train(self,k=1,verbose=0):
        """
        Learning function
        
        Parameters
        ----------
        k=1, int, k in the knn system
        verbose=0, bool, verbosity mode
        
        Returns
        -------
        chart, array of shape (nblearningSamples, rdim) 
               the resulting data embedding
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

    def test(self, x):
        """
        Function to generalize the embedding to new data
        
        Parameters
        ----------
        x: array of shape(nbitems,fdim) 
           new data points to be embedded
        verbose=0, bool, verbosity mode
        
        Returns
        -------
        chart,array of shape (nbitems,rdim) 
                    resulting rdim-dimensional represntation
        """     
        if self.trained ==0:
            raise ValueError, "Untrained function -- cannot generalize"
        if np.size(x)==self.fdim:
            x = np.reshape(x,(1,self.fdim))
        self.check_data(x)
        #
        u = np.dot(x,self.projector)
        return u

#------------------------------------------------------------------
#------- Ancillary methods ----------------------------------------
#------------------------------------------------------------------

def check_isometry(G, chart, nseeds=100, verbose = 0):
    """
    A simple check of the Isometry:
    look whether the output distance match the intput distances
    for nseeds points
    
    Returns
    -------
    a scaling factor between the proposed and the true metrics
    """
    nseeds = np.minimum(nseeds, G.V)
    aux = np.argsort(nr.rand(nseeds))
    seeds =  aux[:nseeds]
    dY = Euclidian_distance(chart[seeds],chart)
    dx = G.floyd(seeds)

    dY = np.reshape(dY,np.size(dY))
    dx = np.reshape(dx,np.size(dx))

    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(dx,dY,'.')
        mp.show()

    scale = np.dot(dx,dY)/np.dot(dx,dx)
    return scale
    


def _sparse_local_correction_for_embedding(G,chart,sigma = 1.0,niter=100):
    """
    WIP : an unfinished fuction that aims at improving isomap's prbs
    the idea is to optimize the representation of local distances
    
    Parameters
    ----------
    G: WeightedGraph instance, 
       the graph to be isomapped
    chart: array of shape (G.V,dim)
           the input chart
    sigma, float a scale parameter
    
    Returns
    -------
    chart : the corrected chart
    
    Note
    ----
    the graph G is reordered
    """
    G.reorder(0)
    sqsigma = 2*sigma**2
    dx = G.weights
    weight = np.exp(-dx**2/sqsigma)

    K = G.copy()
    K.set_euclidian(chart)
    dY = K.weights 

    tiny = 1.e-10
    ln = G.list_of_neighbors()
    ci, ne, we = G.to_neighb()
    
    for i in range(niter):
        aux = weight*(dx-dY)/np.maximum(dY,tiny)        
        grad = np.zeros(np.shape(chart))
        for j in range(G.V):
            k = ln[j]
            grad[j,:] = np.dot(aux[ci[j]:ci[j+1]],
                               np.squeeze(chart[k,:]-chart[j,:]))/np.size(k)

        chart = chart - grad
        K.set_euclidian(chart)
        dY = K.weights
    return chart
    
def partial_floyd_graph(G,k):
    """
    Create a graph of the knn in the geodesic sense, given an input graph G
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

def swiss_roll(nbitem=1000):
    """
    Sample nbitem=1000 point from a swiss roll

    Returns
    -------
    x array of shape (nbitem,3) the 3D embedding
    u array of shape (nbitem,2) the intrinsic coordinates
    """
    u = nr.rand(nbitem,2)
    r = 1+u[:,0]
    x1 = np.reshape(r*(np.cos(2*np.pi*u[:,0])),(nbitem,1))
    x2 = np.reshape(r*(np.sin(2*np.pi*u[:,0])),(nbitem,1))
    x3 = np.reshape(10*u[:,1],(nbitem,1))
    x = np.hstack((x1,x2,x3))
    return x,u

def _orange(nbsamp=1000, k=10):
    """
    Sample points from an 'orange' (a sphere with two partial cuts)
    
    Parameters
    ----------
    nbsamp=1000, int number of points to draw
    k=10, int, number of neighboring points in the graph
    
    Returns
    -------
    G Weighted_Graph instance that represents the meshing of points
    X array of shape(nbsamp,3) the positions of these points 
    """
    # make the sphere
    x = nr.randn(nbsamp,3)
    X = x.T/np.sqrt(np.sum(x**2,1))
    X = X.T
    G = fg.WeightedGraph(nbsamp)
    G.knn(X,k)

    # make the cuts
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
    u = isomap(G,rdim,nbseed,1)
    check_isometry(G,u[:,:2],nseeds  =100)
    
    
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
    x = nr.randn(n,dim)
    G = fg.WeightedGraph(n)
    G.knn(x, 5)
    G.set_gaussian(x)
    LE(G, 5, verbose)

def _test_lE1(verbose=0):
    n = 100
    t = 2*np.pi*nr.randn(n)
    x = np.transpose(np.vstack((np.cos(t),np.sin(t),t)))
    x = x+0.1*nr.randn(x.shape[0],x.shape[1])
    G = fg.WeightedGraph(n)
    G.knn(x,10)
    G.set_gaussian(x)
    u = LE(G,dim=5)
    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(t,u[:,0],'.')
        mp.figure()
        mp.plot(u[:,0],u[:,1],'.')
        mp.figure()
        mp.plot(x[:,0],x[:,1],'.')
    x = G.cc();
    print x.max()

def _test_LPP(verbose=0):
    n = 100
    t = 2*np.pi*nr.randn(n)
    x = np.transpose(np.vstack((np.cos(t),np.sin(t),t)))
    x = x+0.1*nr.randn(x.shape[0],x.shape[1])
    G = fg.WeightedGraph(n)
    G.knn(x,3)
    G.set_gaussian(x)
    u = LPP(G,x,dim=2)
    em = np.dot(x,u)
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
    x = nr.randn(n,dim)#
    for k in range(r):
        x[:,k] += 2*nr.randn(n)

    infer_latent_dim(x,verbose) 

