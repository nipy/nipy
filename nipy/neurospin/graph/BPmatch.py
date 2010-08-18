"""
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
Routines for Matching of a graph to a cloud of points/tree structures through
Bayesian networks (Belief propagation) algorithms 

Author: Bertrand Thirion , 2006-2008.

"""

import numpy as np

import graph as fg
from nipy.neurospin.eda.dimension_reduction import Euclidian_distance


def BPmatch(c1, c2, graph, dmax):
    """
    Matching the rows of c1 to those of c2 based on their relative positions
    
    Parameters
    ----------
    c1, array of shape (nbitems1, dim),
        dataset 1
    c2, array of shape (nbitems2, dim),
        dataset 2
    scale, float, scale parameter
    eps = 1.e-12, float, 

    Returns
    -------
    i, j, k: arrays of shape(E) 
             sparse adjacency matrix of the bipartite association graph
    """ 
    belief = fg.graph_bpmatch(c1, c2, graph, dmax)
    i,j = np.where(belief);
    k = belief[i, j]
    return i, j, k

def match_trivial(c1, c2, scale, eps = 1.e-12 ):
    """
    Matching the rows of c1 to those of c2 based on their relative positions
    
    Parameters
    ----------
    c1, array of shape (nbitems1, dim),
        dataset 1
    c2, array of shape (nbitems2, dim),
        dataset 2
    scale, float, scale parameter
    eps = 1.e-12, float, 

    Returns
    -------
    i, j, k: arrays of shape(E) 
           sparse adjacency matrix of the bipartite association graph
    """
    sqs = 2*scale**2
    
    # make a prior
    D = Euclidian_distance(c1,c2)
    W = np.exp(-D*D/sqs);
    W = W*(D<3*scale);
    sW = np.sum(W,1)
    if np.sum(sW)<eps:
        return np.array([]),np.array([]),np.array([])
    W = (W.T/np.maximum(eps,np.sum(W,1))).T
    
    i,j = np.where(W);
    k = W[i,j]
    return i,j,k

def BPmatch_slow_asym_dev(c1, c2, G1, G2, scale):
    """
    New version which makes the differences between ascending
    and descending links
    
    Parameters
    ----------
    c1, c2 are arrays of shape (n1,d) and (n2,d) that represent
       features or coordinates ,
       where n1 and n2 are the number of things to be put in correpondence
       and d is the common dimension
    G1 and G2 are corresponding graphs (forests in fff sense)
    scale is  a typical distance to compare positions
    
    Returns
    -------
    (i,j,k): sparse model of the probabilistic relationships,
             where k is the probability that i is associated with j
    """
    if G1.V != c1.shape[0]:
        raise ValueError, "incompatible dimension for G1 and c1"

    if G2.V != c2.shape[0]:
        raise ValueError, "incompatible dimension for G2 and c2"

    sqs = 2*scale*scale
    ofweight = np.exp(-0.5)

    # get the distances
    D = Euclidian_distance(c1,c2)
    W = np.exp(-D*D/sqs)

    # add an extra default value and normalize
    W = np.hstack((W,ofweight*np.ones((G1.V,1))))
    W = (W.T/np.sum(W,1)).T

    W = _MP_algo_dev_(G1,G2,W,c1,c2,sqs)
    W = W[:,:-1]
    
    # find the leaves of the graphs 
    sl1 = G1.isleaf().astype('bool')
    sl2 = G2.isleaf().astype('bool')
    # cancel the weigts of non-leaves
    W[sl1==0,:]=0
    W[:,sl2==0]=0

    W[W<1.e-4]=0
    i,j = np.where(W)
    k = W[i,j]
    return i,j,k


def _son_translation_(G,c):
    """
    Given a forest strcuture G and a set of coordinates c,
    provides the coorsdinate difference ('translation')
    associated with each descending link
    """
    v = np.zeros((G.E,c.shape[1]))
    for e in range(G.E):
        if G.weights[e]<0:
            ip = G.edges[e,0]
            target = G.edges[e,1]
            v[e,:] = c[target,:]-c[ip,:]        
    return v


def _MP_algo_(G1,G2,W,c1,c2,sqs,imax= 100, eps = 1.e-12 ):
    """
    """
    eps = eps
    #get the graph structure
    i1 = G1.get_edges()[:,0]
    j1 = G1.get_edges()[:,1]
    k1 = G1.get_weights()
    i2 = G2.get_edges()[:,0]
    j2 = G2.get_edges()[:,1]
    k2 = G2.get_weights()
    E1 = G1.E
    E2 = G2.E

    # define vectors related to descending links v1,v2
    v1 = _son_translation_(G1,c1)
    v2 = _son_translation_(G2,c2)   

    # the variable tag is used to reduce the computation load...
    if E1>0:
        tag = G1.converse_edge()

        # make transition matrices
        TM = []
        for e in range(E1):
            if k1[e]>0:
                # ascending links
                t = eps*np.eye(G2.V)
                
                for f in range(E2):
                    if k2[f]<0:
                        du = v1[tag[e],:]-v2[f,:]
                        nuv = np.sum(du**2)
                        t[j2[f],i2[f]] = np.exp(-nuv/sqs)
                        
            if k1[e]<0:
                #descending links
                t = eps*np.eye(G2.V)
                
                for f in range(E2):
                    if k2[f]<0:
                        du = v1[e,:]-v2[f,:]
                        nuv = np.sum(du**2)
                        t[i2[f],j2[f]] = np.exp(-nuv/sqs)
                
            t = (t.T/np.maximum(eps,np.sum(t,1))).T
            
            TM.append(t)

        # the BP algo itself
        # init the messages
        M = np.zeros((E1,G2.V)).astype('d')
        Pm = W[i1,:]
        B = W.copy()
        
        now = float(G1.V) 
        for iter in range(imax):
            Bold = B.copy()
            B = W.copy()
            
            # compute the messages
            for e in range(E1):
                t = TM[e]
                M[e,:] = np.dot(Pm[e,:],t)

            sM = np.sum(M,1)
            sM = np.maximum(sM,eps)
            M = (M.T/sM).T
                        
            # update the beliefs
            for e in range(E1):
                B[j1[e],:] = B[j1[e],:]*M[e,:]

            B = (B.T/np.maximum(eps, np.sum(B,1))).T
            dB = np.sqrt(((B-Bold)**2).sum())
            if dB<eps*now:
                 # print dB/now,iter
                 break
                        
            #prepare the next message
            for e in range(E1):
                me = np.maximum(eps, M[tag[e],:])
                Pm[e,:] = B[i1[e],:]/me
    else:
        B = W

    B = (B.T/np.maximum(eps,np.sum(B,1))).T
    
    return B


def _MP_algo_dev_(G1, G2, W, c1, c2, sqs, imax=100, eta=1.e-6 ):
    """
    Internal part of the graph matching procedure.
    Not to be read by normal people.

    Parameters
    ----------
    G1 and G2: Forests instance
               these describe the internal 
    W: array of shape(n1, n2),
       initial correpondence matrix
    c1, c2: arrays of shape (n1,d) and (n2,d)
            features or coordinates to be matched,
            where n1, n2 = number of things to be put in correpondence
            and d = common dimension
    sqs: typical distance to compare positions
    imax: int,
          maximal number of iterations         
    eta: float, optional,
         a regularization constant <<1 that avoids inconsistencies

    Returns
    -------
    W: array of shape(n1, n2),
       updated correspondence matrix
    """
    eps = 1.e-12

    #get the graph structure
    E1 = G1.E
    E2 = G2.E
    if E1<1: return W
    if E2<1: return W
    i1 = G1.get_edges()[:,0]
    j1 = G1.get_edges()[:,1]
    k1 = G1.get_weights()
    i2 = G2.get_edges()[:,0]
    j2 = G2.get_edges()[:,1]
    k2 = G2.get_weights()

    # define vectors related to descending links v1,v2
    v1 = _son_translation_(G1,c1)
    v2 = _son_translation_(G2,c2)   

    # the variable tag is used to reduce the computation load...
    tag = G1.converse_edge().astype(np.int)
    
    # make transition matrices
    TM = []
    for e in range(E1):
        if k1[e]>0:
            # ascending links
            t = eta*np.ones((G2.V+1,G2.V+1))
            for f in range(E2):
                if k2[f]<0:
                    #du = v1[tag[e],:]-v2[f,:]
                    #nuv = np.sum(du**2)
                    t[j2[f],i2[f]] = 1
                        
        if k1[e]<0:
            #descending links
            t = eta*np.ones((G2.V+1,G2.V+1))
            for f in range(E2):
                if k2[f]<0:
                    du = v1[e,:]-v2[f,:]
                    nuv = np.sum(du**2)
                    t[i2[f],j2[f]] = np.exp(-nuv/sqs)
                
        t = (t.T/np.maximum(eps,np.sum(t,1))).T
        TM.append(t)

    # the BP algo itself
    # init the messages
    M = np.zeros((E1,G2.V+1)).astype('d')
    Pm = W[i1,:]
    B = W.copy()
    now = float(G1.V) 
        
    for iter in range(imax):
        Bold = B.copy()
        B = W.copy()
                    
        # compute the messages
        for e in range(E1):
            t = TM[e]
            M[e,:] = np.dot(Pm[e,:],t)

        sM = np.sum(M,1)
        sM = np.maximum(sM,eps)
        M = (M.T/sM).T
                        
        # update the beliefs
        for e in range(E1):
            B[j1[e],:] = B[j1[e],:]*M[e,:]

        B = (B.T/np.maximum(eps,np.sum(B,1))).T
        dB = np.sqrt(((B-Bold)**2).sum())
        if dB<eps*now:
            # print dB/now,iter
            break

        #prepare the next message
        for e in range(E1):
            me = np.maximum(eps,M[tag[e],:])
            Pm[e,:] = B[i1[e],:]/me

    B = (B.T/np.maximum(eps,np.sum(B,1))).T
    return B
    
