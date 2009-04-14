"""
These routines perform some hierrachical agglomerative clustering
of some input data. The following alternatives are proposed:
- Distance based average-link
- Similarity-based average-link
- Distance based maximum-link
- Ward's algorithm under graph constraints
- Ward's algorithm without graph constraints

In this latest version, the results are returned in a 'WeightedForest'
structure, which gives access to the clustering hierarchy, facilitates
the plot of the result etc.

For back-compatibility, *_segment versions of the algorithms have been
appended, with the old API (except the qmax parameter, which now
represents the number of wanted clusters)

Author : Bertrand Thirion,Pamela Guevara, 2006-2009
"""

#---------------------------------------------------------------------------
# ------ Routines for Agglomerative Hierarchical Clustering ----------------
# --------------------------------------------------------------------------

import numpy as np
import nipy.neurospin.graph.graph as fg
from nipy.neurospin.eda.dimension_reduction import Euclidian_distance
from nipy.neurospin.clustering.clustering import ward

class WeightedForest(fg.Forest):
    """
    This is a weighted Forest structure, i.e. a tree
    - ecah node has one parent and children
    (hierarchical structure)
    - some of the nodes can be viewed as leaves, other as roots
    - the edges within a tree are associated with a weight:
    +1 from child to parent
    -1 from parent to child
    - additionally, the nodes have a value, which is called 'height',
    especially useful from dendrograms

    fields:
    - V : (int,>0) the number of vertices
    - E : (int) the number of edges
    - parents: array of shape (self.V) the parent array
    - edges: array of shape (self.E,2) reprensenting pairwise neighbors
    - weights, array of shape (self.E), +1/-1 for scending/descending links 
    - children: list of arrays that represents the childs of any node
    - height: array of shape(self.V)
    """
    def __init__(self, V, parents=None,height=None):
        """
        INPUT:
        V: the number of edges of the graph
        parents = None: array of shape (V) the parents of the graph
        by default, the parents are set to range(V), i.e. each  
        node is its own parent, and each node is a tree
        height=None: array of shape(V) the height of the nodes
        """
        V = int(V)
        if V<1:
            raise ValueError, 'cannot create graphs with no vertex'
        self.V = int(V)

        # define the parents
        if parents==None:
            self.parents = np.arange(self.V)
        else:
            if np.size(parents)!=V:
                raise ValueError, 'Incorrect size for parents'
            if parents.max()>self.V:
                raise ValueError, 'Incorrect value for parents'             
            self.parents = np.reshape(parents,self.V)

        self.define_graph_attributes()

        if self.check()==0:
            raise ValueError, 'The proposed structure is not a forest'
        self.children = []

        if height==None:
            height=np.zeros(self.V)
        else:
            if np.size(height)!=V:
                raise ValueError, 'Incorrect size for height'   
            self.height = np.reshape(height,self.V)

    def set_height(self,height=None):
        """
        set the height array
        """
        if height==None:
            height = np.zeros(self.V)
            
        if np.size(height)!=self.V:
            raise ValueError, 'Incorrect size for height'   

        self.height = np.reshape(height,self.V)

    def get_height(self):
        """
        get the height array
        """
        return self.height
    
    def check_compatible_height(self):
        """
        Check that height[parents[i]]>=height[i] for all nodes
        """
        OK = True
        for i in range(self.V):
            if self.height[self.parents[i]]<self.height[i]:
                OK = False
        return OK

    def plot(self):
        """
        Plot the dendrogram associated with self
        the rank of the data in the dendogram is returned
        """
        if self.check_compatible_height()==False:
            raise ValueError, 'cannot plot myself in my current state'

        n = np.sum(self.isleaf())

        # 1. find a permutation of the leaves that makes it nice
        aux = _label(self.parents)
        temp = np.zeros(self.V)
        rank = np.arange(self.V)
        temp[:n] = np.argsort(aux[:n])
        for i in range(n):
            rank[temp[i]]=i

        # 2. derive the abscissa in the dendrogram
        idx = np.zeros(self.V)
        temp = np.argsort(rank[:n])
        for i in range(n):
            idx[temp[i]]=i
        for i in range(n,self.V):
            j = np.nonzero(self.parents==i)[0]
            idx[i] = np.mean(idx[j])

        # 3. plot
        import matplotlib.pylab as mp
        mp.figure()

        for i in range(self.V):
            h1 = self.height[i]
            h2 = self.height[self.parents[i]]
            mp.plot([idx[i],idx[i]],[h1,h2] ,'k')

        ch = self.get_children()    
        for i in range(self.V):
            if np.size(ch[i])>0:
                lidx = idx[ch[i]]
                m = lidx.min()
                M = lidx.max()
                h = self.height[i]
                mp.plot([m,M],[h,h],'k')
                
        cM = 1.05*self.height.max()-0.05*self.height.min()
        cm = 1.05*self.height.min()-0.05*self.height.max()
        mp.axis([-1,idx.max()+1,cm,cM])
        mp.show()
        return rank

    def partition(self,threshold):
        """
        partition the tree according to a cut criterion
        """
        valid = self.height<threshold
        f = self.subforest(valid)
        u = f.cc()
        return u[f.isleaf()]

    def split(self,k):
        """
        idem as partition, but a number of components are supplied instead
        """
        k = int(k)
        if k>self.V: k = self.V
        nbcc = self.cc().max()+1
    
        if k<=nbcc:
            u = self.cc()
            return u[self.isleaf()]
        
        sh = np.sort(self.height)
        th = sh[-(k-nbcc)]
        u = self.partition(th)
        return u

    def plot_height(self):
        """
        plot the height of the non-leaves nodes
        """
        import matplotlib.pylab as mp
        mp.figure()
        sh = np.sort(self.height[self.isleaf()==False])
        n = np.sum(self.isleaf()==False)
        mp.bar(np.arange(n),sh)
        mp.show()

    def list_of_subtrees(self):
        """
        returns the list of all non-trivial subtrees in the graph
        Caveat: theis function assumes that the vertices are sorted in a
        way such that parent[i]>i forall i
        Only the leaves are listeed, not the subtrees themselves
        """
        lst = []
        n = np.sum(self.isleaf())
        for i in range(self.V):
            lst.append(np.array([], np.int))
        for i in range(n):
            lst[i] = np.array([i], np.int)
        for i in range(self.V-1):
            j = self.parents[i]
            lst[j] = np.hstack((lst[i],lst[j]))

        return lst[n:self.V]
        
    def plot2(self, addNodes = False, font_size = 10, cl_size = None):
        """
        Plot the dendrogram associated with self
        """
        if self.check_compatible_height()==False:
            raise ValueError, 'cannot plot myself in my current state'
        n = np.sum(self.isleaf())
        
        # 1. find a permutation of the leaves that makes it nice
        aux = _label2(self.parents, self.children)
        rank = aux

        # 2. derive the abscissa in the dendrogram
        idx = -np.ones(self.V)
        temp2 = np.argsort(rank[:self.V])
        isleaf = self.isleaf()
        temp = np.zeros(n)
        notleaf = np.zeros(self.V-n)
        nleaf = []
        
        cont = 0
        cont2 = 0
        for i in xrange( np.size(rank) ):
            ind = temp2[i]
            if isleaf[ind]:
                temp[cont] = temp2[i]
                cont = cont + 1
            else:
                notleaf[cont2] = temp2[i]
                nleaf.append(int(temp2[i]))
                cont2 = cont2 + 1
        
        for i in range(n):
            idx[temp[i]]=i
        
        while len(nleaf) > 0:
            for i in nleaf:
                ch = self.children[i]
                idxch = idx[ch]
                test = idxch == -1
                ok = True
                for t in test:
                    if t == True:
                        ok = False
                if ok:
                    idx[i] = np.mean(idxch)
                    nleaf.remove(i)
    
        # 3. plot
        import matplotlib.pylab as mp
        mp.figure()

        for i in range(self.V):
            h1 = self.height[i]
            h2 = self.height[self.parents[i]]
            mp.plot([idx[i],idx[i]],[h1,h2] ,'k')

        ch = self.get_children()    
        for i in range(self.V):
            if np.size(ch[i])>0:
                lidx = idx[ch[i]]
                m = lidx.min()
                M = lidx.max()
                h = self.height[i]
                mp.plot([m,M],[h,h],'k')
                
        cM = 1.05*self.height.max()-0.05*self.height.min()
        cm = 1.05*self.height.min()-0.05*self.height.max()
        mp.axis([-1,idx.max()+1,cm,cM])
        mp.show()
        
        if addNodes:
            for i in range(self.V):
                h1 = self.height[i]
                mp.text(idx[i]+0.05, h1+0.45, str(i), fontsize = font_size, color = 'b')
        
        if cl_size != None:
            for i in range(self.V):
                h1 = self.height[i]
                text = str(cl_size[i])
                mp.text(idx[i]+0.5, h1+0.05, text, fontsize = font_size)
        
        mp.show()
        return rank

    def fancy_plot_(self,valid):
        """
        Idem plot, but the valid edges are enahanced
        """
        if self.check_compatible_height()==False:
            raise ValueError, 'cannot plot myself in my current state'

        n = np.sum(self.isleaf())  
        # 1. find a permutation of the leaves that makes it nice
        aux = _label(self.parents)
        temp = np.zeros(self.V)
        rank = np.arange(self.V)
        temp[:n] = np.argsort(aux[:n])
        for i in range(n):
            rank[temp[i]]=i

        # 2. derive the abscissa in the dendrogram
        idx = np.zeros(self.V)
        temp = np.argsort(rank[:n])
        for i in range(n):
            idx[temp[i]]=i
        for i in range(n,self.V):
            j = np.nonzero(self.parents==i)[0]
            idx[i] = np.mean(idx[j])

        # 3. plot
        import matplotlib.pylab as mp
        mp.figure()

        for i in range(self.V):
            h1 = self.height[i]
            h2 = self.height[self.parents[i]]
            mp.plot([idx[i],idx[i]],[h1,h2] ,'k')
            if valid[i]&valid[self.parents[i]]:
                mp.plot([idx[i],idx[i]],[h1,h2] ,'b',linewidth=2)

        ch = self.get_children()    
        for i in range(self.V):
            if np.size(ch[i])>0:
                chi = ch[i]
                lidx = idx[chi]
                j = chi[lidx.argmin()]
                k = chi[lidx.argmax()]
                m = lidx.min()
                M = lidx.max()
                h = self.height[i]
                mp.plot([m,M],[h,h],'k')
                if valid[j]|valid[k]:
                    mp.plot([m,M],[h,h],'b',linewidth=2)
                
        cM = 1.05*self.height.max()-0.05*self.height.min()
        cm = 1.05*self.height.min()-0.05*self.height.max()
        mp.axis([-1,idx.max()+1,cm,cM])
        mp.show()
        return rank
    
    def fancy_plot(self,validleaves):
        """
        Idem plot, but the valid edges are enahanced
        """
        if self.check_compatible_height()==False:
            raise ValueError, 'cannot plot myself in my current state'

        n = np.sum(self.isleaf())
        valid = np.zeros(self.V,'bool')
        valid[self.isleaf()]=validleaves.astype('bool')
        nv =  np.sum(validleaves)
        nv0 = 0
        while nv>nv0:
            nv0= nv
            for v in range(self.V):
                if valid[v]:
                    valid[self.parents[v]]=1
            nv = np.sum(valid)
            
        rank = self.fancy_plot_(valid)
        return rank



#--------------------------------------------------------------------------
#------------- Average link clustering ------------------------------------
# -------------------------------------------------------------------------


def Average_Link_Euclidian(X,verbose=0):
    """
    Average link clustering based on data matrix.
    INPUT:
    - X: a (nbitem,dim) array from which an Euclidian distance
    matrix is computed
    - verbose=0, verbosity level
    OUTPUT:
    -t a weightForest structure that represents the dendrogram of the data
    NOTE:
    this method has not been optimized
    """
    if X.shape[0]==np.size(X):
        X = np.reshape(X,(np.size(X),1))
    if np.size(X)<10000:
        D = Euclidian_distance(X)
    else:
        raise ValueError, "The distance matrix is too large"
    
    t = Average_Link_Distance(D,verbose)
    return t 


def Average_Link_Distance(D,verbose=0):
    """
    Average link clustering based on a pairwise distance matrix.

    Average_Link_Distance(D,verbose=0):
    INPUT:
    - D: a (n,n) distance matrix between some items
    - verbose=0, verbosity level
    OUTPUT:
    -t a weightForest structure that represents the dendrogram of the data
    NOTE:
    this method has not been optimized
    """

    n = D.shape[0]
    if D.shape[1]!=n:
        raise ValueError, "non -square distance matrix" 
    
    DI = np.infty*np.ones((2*n,2*n))
    DI[:n,:n]=D+np.diag(np.infty*np.ones(n))

    parent = np.arange(2*n-1, dtype=np.int)
    pop = np.ones(2*n-1, dtype=np.int)
    height = np.zeros(2*n-1)
    
    for q in range(n-1):
        d = DI.min()
        k = q+n
        height[k] = d   
        i,j = np.where(DI==d)
        i = i[0]
        j = j[0]
        parent[i] = k
        parent[j] = k
        pop[k] = pop[i]+pop[j]
        DI[k] = (DI[i]*pop[i]+DI[j]*pop[j])/pop[k]
        DI[:,k] = np.transpose(DI[k,:])
        DI[i,:] = np.infty
        DI[j,:] = np.infty
        DI[:,i] = np.infty
        DI[:,j] = np.infty

    t = WeightedForest(2*n-1,parent,height)

    return t


def Average_Link_Distance_segment(D,stop=-1,qmax=1,verbose=0):
    """
    Average link clustering based on a pairwise distance matrix.

    Average_Link_Distance(D,stop=-1,qmax=-1,verbose=0):
    INPUT:
    - D: a (n,n) distance matrix between some items
    - stop=-1: stopping criterion, i.e. distance threshold at which
    further merges are forbidden
    By default, all merges are performed
    - qmax = 1; the number of desired clusters
    (in the limit of stop)
    - verbose=0, verbosity level
    OUTPUT:
    -u: a labelling of the graph vertices according to the criterion
    -cost the cost of each merge step during the clustering procedure
    NOTE:
    this method has not been optimized
    """

    n = D.shape[0]
    if D.shape[1]!=n:
        raise ValueError, "non -square distance matrix" 
    if stop==-1: stop = np.infty
    

    t = Average_Link_Distance(D,verbose)
    if verbose: t.plot()

    u1 = np.zeros(n, np.int)
    u2 = np.zeros(n, np.int)
    if stop>=0:
        u1 = t.partition(stop)
    if qmax>0:
        u2 = t.split(qmax)

    if u1.max()<u2.max():
        u = u2
    else:
        u = u1

    cost = t.get_height()
    cost = cost[t.isleaf()==False]
    
    return u,cost

#--------------------------------------------------------------------------
#------------- Average link clustering on a graph -------------------------
# -------------------------------------------------------------------------

def fusion(K,pop,i,j,k):
    """
    fusion(K,pop,i,j,k)
    modifies the graph K to merge nodes i and  j into nodes k
    the similarity values are weighted averaged, where pop[i] and pop[j]
    yield the relative weights.
    this is used in average_link_slow (deprecated)
    """
    #
    fi = float(pop[i])/(pop[k])
    fj = 1.0-fi
    #
    # replace i ny k
    #
    idxi = np.nonzero(K.edges[:,0]==i)
    np.reshape(idxi,np.size(idxi))
    K.weights[idxi] = K.weights[idxi]*fi
    K.edges[idxi,0] = k
    idxi = np.nonzero(K.edges[:,1]==i)
    np.reshape(idxi,np.size(idxi))
    K.weights[idxi] = K.weights[idxi]*fi
    K.edges[idxi,1] = k
    #
    # replace j by k
    #
    idxj = np.nonzero(K.edges[:,0]==j)
    np.reshape(idxj,np.size(idxj))
    K.weights[idxj] = K.weights[idxj]*fj
    K.edges[idxj,0] = k
    idxj = np.nonzero(K.edges[:,1]==j)
    np.reshape(idxj,np.size(idxj))
    K.weights[idxj] = K.weights[idxj]*fj
    K.edges[idxj,1] = k
    #
    #sum/remove double edges
    #
    #left side
    idxk = np.nonzero(K.edges[:,0]==k)[0]
    corr = K.edges[idxk,1]
    scorr = np.sort(corr)
    acorr = np.argsort(corr)
    for a in range(np.size(scorr)-1):
        if scorr[a]==scorr[a+1]:
            i1 = idxk[acorr[a]]
            i2 = idxk[acorr[a+1]]
            K.weights[i1] =  K.weights[i1]+K.weights[i2]
            K.weights[i2] = -np.infty
            K.edges[i2,:] = -1
            
    #right side
    idxk = np.nonzero(K.edges[:,1]==k)[0]
    corr = K.edges[idxk,0]
    scorr = np.sort(corr)
    acorr = np.argsort(corr)
    for a in range(np.size(scorr)-1):
        if scorr[a]==scorr[a+1]:
            i1 = idxk[acorr[a]]
            i2 = idxk[acorr[a+1]]
            K.weights[i1] =  K.weights[i1]+K.weights[i2]
            K.weights[i2] = -np.infty
            K.edges[i2,:] = -1

def Average_Link_Graph(G):
    """
    Agglomerative function based on a (hopefully sparse) similarity graph
    INPUT:
    - G the input graph
    OUPUT:
    - t a weightForest structure that represents the dendrogram of the data
    CAVEAT:
    In that case, the homogeneity is associated with high similarity
    (as opposed to low cost as in most clustering procedures,
    e.g. distance-based procedures).  Thus the tree is created with
    negated affinity values, in roder to respect the traditional
    ordering of cluster potentials. individual points have the
    potential (-np.infty).
    This problem is handled transparently inthe associated segment functionp.
    """

    # prepare a graph with twice the number of vertices
    n = G.V
    nbcc = G.cc().max()+1
    
    K = fg.WeightedGraph(2*G.V)
    K.E = G.E
    K.edges = G.edges.copy()
    K.weights = G.weights.copy()
    
    parent = np.arange(2*n-nbcc, dtype=np.int)
    pop = np.ones(2*n-nbcc, np.int)
    height = np.infty*np.ones(2*n-nbcc)

    # iteratively merge clusters
    for q in range(n-nbcc):

        # 1. find the heaviest edge
        m = (K.weights).argmax()
        cost = K.weights[m]
        k = q+n
        height[k] = cost
        i = K.edges[m,0]
        j = K.edges[m,1]
        
        # 2. remove the current edge
        K.edges[m,:] = -1
        K.weights[m] = -np.infty
        m = np.nonzero((K.edges[:,0]==j)*(K.edges[:,1]==i))[0]
        K.edges[m,:] = -1
        K.weights[m] = -np.infty
        
        # 3. merge the edges with third part edges
        parent[i] = k
        parent[j] = k
        pop[k] = pop[i]+pop[j]
        fusion(K,pop,i,j,k)

    height[height<0]=0
    height[np.isinf(height)] = height[n]+1
    t = WeightedForest(2*n-nbcc,parent,-height)

    return t

def Average_Link_Graph_segment(G,stop=0,qmax=1,verbose=0):
    """
    Agglomerative function based on a (hopefully sparse) similarity graph
    INPUT:
    - G the input graph
    - stop = 0: the stopping crterion
    - qmax=1: the number of desired clusters
    (in the limit of the stopping criterion)
    OUPUT:
    -u: a labelling of the graph vertices according to the criterion
    -cost the cost of each merge step during the clustering procedure
    """

    # prepare a graph with twice the number of vertices
    n = G.V
    if qmax==-1: qmax = n
    qmax = int(np.minimum(qmax,n))

    t = Average_Link_Graph(G)

    if verbose: t.plot()

    u1 = np.zeros(n, np.int)
    u2 = np.zeros(n, np.int)
    if stop>=0:
        u1 = t.partition(-stop)
    if qmax>0:
        u2 = t.split(qmax)

    if u1.max()<u2.max():
        u = u2
    else:
        u = u1

    cost = -t.get_height()
    cost = cost[t.isleaf()==False]
    #cost = cost[:u.max()]
    
    return u,cost



#--------------------------------------------------------------------------
#------------- Ward's algorithm with graph constraints --------------------
# -------------------------------------------------------------------------



def _inertia(i,j,Features):
    """
    Compute the variance of the set which is
    the concatenation of Feature[i] and Features[j]
    """
    if np.size(np.shape(Features[i]))<2:
        print i, np.shape(Features[i]),Features[i]
    if np.size(np.shape(Features[i]))<2:
        print j, np.shape(Features[j]),Features[j]
    if np.shape(Features[i])[1]!=np.shape(Features[j])[1]:
        print i,j,np.shape(Features[i]), np.shape(Features[j])
    localset = np.vstack((Features[i],Features[j]))
    return np.var(localset,0).sum()

def _initial_inertia(K,Features,seeds=None):
    """
    Compute the variance associated with each
    edge-related pair of vertices
    Thre sult is written in K;weights
    if seeds if provided (seeds!=None)
    this is done only for vertices adjacent to the seeds
    """
    if seeds==None:
        for e in range(K.E):
            i = K.edges[e,0]
            j = K.edges[e,1]
            ESS = _inertia(i,j,Features)
            K.weights[e] = ESS
    else:
        aux = np.zeros(K.V).astype('bool')
        aux[seeds]=1
        for e in range(K.E):
            i = K.edges[e,0]
            j = K.edges[e,1]
            if (aux[i] or aux[j]):
                K.weights[e] = _inertia(i,j,Features)
            else:       
                K.weights[e] = np.infty

def _auxiliary_graph(G,Features):
    """
    prepare a graph with twice the number of vertices
    this graph will contain the connectivity information
    along the merges.
    """
    K = fg.WeightedGraph(2*G.V-1)
    K.E = G.E
    K.edges = G.edges.copy()
    K.weights = np.zeros(K.E)
    #
    K.symmeterize()
    valid = K.edges[:,0]<K.edges[:,1]
    K.remove_edges(valid)
    #
    K.remove_trivial_edges()
    _initial_inertia(K,Features)
    return K

def _remap(K,i,j,k,Features,linc,rinc):
    """
    K,inc,rinc = _remap_dev(K,i,j,k)
    modifies the graph K to merge nodes i and  j into nodes k
    the graph weights are modified accordingly
    """
    # -------
    # replace i by k
    # --------
    idxi = np.array(linc[i]).astype(np.int)
    if np.size(idxi)>1:
        for l in idxi:
            K.weights[l] = _inertia(k,K.edges[l,1],Features)
    elif np.size(idxi==1):
        K.weights[idxi] = _inertia(k,K.edges[idxi,1],Features)
    if np.size(idxi)>0:
        K.edges[idxi,0] = k
    
    idxi = np.array(rinc[i]).astype(np.int)
    if np.size(idxi)>1:
        for l in idxi :
            K.weights[l] = _inertia(K.edges[l,0],k,Features)
    elif np.size(idxi)==1:
        K.weights[idxi] = _inertia(K.edges[idxi,0],k,Features)
    if np.size(idxi)>0:
        K.edges[idxi,1] = k

    #------
    # replace j by k
    #------- 
    idxj = np.array(linc[j]).astype(np.int)
    if np.size(idxj)>1:
        for l in idxj :
            K.weights[l] = _inertia(k,K.edges[l,1],Features)
    elif np.size(idxj)==1:
        K.weights[idxj] = _inertia(k,K.edges[idxj,1],Features)
    if np.size(idxj)>0:
        K.edges[idxj,0] = k
    
    idxj = np.array(rinc[j]).astype(np.int)
    if np.size(idxj)>1:
        for l in idxj : 
            K.weights[l] = _inertia(k,K.edges[l,0],Features)
    elif np.size(idxj)==1:
        K.weights[idxj] = _inertia(k,K.edges[idxj,0],Features)
    if np.size(idxj)>0:
        K.edges[idxj,1] = k

    #------
    # update linc,rinc
    #------
    lidxk = list(np.concatenate((linc[j],linc[i])))
    for l in lidxk:
        if K.edges[l,1]==-1:
            lidxk.remove(l)
        
    linc[k]=lidxk
    linc[i] = []
    linc[j] = []
    ridxk = list(np.concatenate((rinc[j],rinc[i])))
    for l in ridxk:
        if K.edges[l,0]==-1:
            ridxk.remove(l)
        
    rinc[k] = ridxk
    rinc[i] = []
    rinc[j] = []

    #------
    #remove double edges
    #------
    #left side
    idxk = np.array(linc[k]).astype(np.int)
    if np.size(idxk)>0:
        corr = K.edges[idxk,1]
        scorr = np.sort(corr)
        acorr = np.argsort(corr)
        for a in range(np.size(scorr)-1):
            if scorr[a]==scorr[a+1]:
                i2 = idxk[acorr[a+1]]
                K.weights[i2] = np.infty
                rinc[K.edges[i2,1]].remove(i2)
                K.edges[i2,:] = -1
                linc[k].remove(i2)
            
    #right side
    idxk = np.array(rinc[k]).astype(np.int)
    if np.size(idxk)>0:
        corr = K.edges[idxk,0]
        scorr = np.sort(corr)
        acorr = np.argsort(corr)
        for a in range(np.size(scorr)-1):
            if scorr[a]==scorr[a+1]:
                i2 = idxk[acorr[a+1]]
                K.weights[i2] = np.infty
                linc[K.edges[i2,0]].remove(i2)
                K.edges[i2,:] = -1
                rinc[k].remove(i2)
    return linc,rinc


def Ward_quick(G,feature,verbose = 0):
    """
    Agglomerative function based on a topology-defining graph
    and a feature matrix. 
    INPUT:
    - G the input graph (a topological graph essentially)
    - feature (G.V,dim_feature) array that yields some vectorial information related to the graph vertices
    OUPUT:
    - t a weightForest structure that represents the dendrogram of the data
    NOTE:
    - Hopefully a quicker version
    - A euclidean distance is used in the feature space
    CAVEAT : only approximate
    """
    # basic check
    if np.size(feature)==feature.shape[0]:
        feature = np.reshape(feature, np.size(feature),1)

    if feature.shape[0]!=G.V:
        raise ValueError, "Incompatible dimension for the feature matrix and the graph"
    
    Features = []
    for i in range(G.V):
        Features.append(np.reshape(feature[i],(1,feature.shape[1])))
    
    n = G.V
    nbcc = G.cc().max()+1
    
    # prepare a graph with twice the number of vertices
    K = _auxiliary_graph(G,Features)
    
    parent = np.arange(2*n-nbcc).astype(np.int)
    pop = np.ones(2*n-nbcc).astype(np.int)
    height = np.zeros(2*n-nbcc)
    linc = K.left_incidence()
    rinc = K.right_incidence()

    # iteratively merge clusters
    q = 0
    while (q<n-nbcc):
        # 1. find the lightest edges
        aux = np.zeros(2*n)
        ape = np.nonzero(K.weights<np.infty)
        ape = np.reshape(ape,np.size(ape))
        idx = np.argsort(K.weights[ape])
    
        for e in range(n-nbcc-q):
            i,j = K.edges[ape[idx[e]],0],K.edges[ape[idx[e]],1]
            if aux[i]==1: break
            if aux[j]==1: break
            aux[i]=1
            aux[j]=1

        emax = np.maximum(e,1)
                
        for e in range(emax):
            m = ape[idx[e]]
            cost = K.weights[m]
            k = q+n
            #if K.weights[m]>=stop: break
            i = K.edges[m,0]
            j = K.edges[m,1]
            height[k] = cost
            if verbose: print q,i,j, m,cost
        
            # 2. remove the current edge
            K.edges[m,:] = -1
            K.weights[m] = np.infty
            linc[i].remove(m)
            rinc[j].remove(m)
            
            ml = linc[j]
            if np.sum(K.edges[ml,1]==i)>0:
                m = ml[np.flatnonzero(K.edges[ml,1]==i)]
                K.edges[m,:] = -1
                K.weights[m] = np.infty
                linc[j].remove(m)
                rinc[i].remove(m)
            
            # 3. merge the edges with third part edges
            parent[i] = k
            parent[j] = k
            totalFeatures = np.vstack((Features[i],Features[j]))
            Features.append(totalFeatures)
            Features[i] = []
            Features[j] = []

            linc,rinc = _remap(K,i,j,k,Features,linc,rinc)
            q+=1

    # build a tree to encode the results
    t = WeightedForest(2*n-nbcc,parent,height)
    
    return t

def Ward_quick_segment(G,feature,stop=-1,qmax=1,verbose = 0):
    """
    Agglomerative function based on a topology-defining graph
    and a feature matrix. 
    INPUT:
    - G the input graph (a topological graph essentially)
    - feature (G.V,dim_feature) array that yields some vectorial information related to the graph vertices
    - stop = -1: the stopping crterion
    if stop==-1, then no stopping criterion is used
    - qmax=1: the maximum number of desired clusters
    (in the limit of the stopping criterion)
    OUPUT:
    -u: a labelling of the graph vertices according to the criterion
    -cost the cost of each merge step during the clustering procedure
    NOTE:
    - Hopefully a quicker version
    - A euclidean distance is used in the feature space
    CAVEAT : only approximate
    """
    # basic check
    if np.size(feature)==feature.shape[0]:
        feature = np.reshape(feature, np.size(feature),1)

    if feature.shape[0]!=G.V:
        raise ValueError, "Incompatible dimension for the feature matrix and the graph"
    

    n = G.V
    if stop==-1: stop = np.infty    
    qmax = int(np.minimum(qmax,n-1))
    nbcc = G.cc().max()+1
    t = Ward_quick(G,feature,verbose)
    if verbose: t.plot()

    u1 = np.zeros(n, np.int)
    u2 = np.zeros(n, np.int)
    if stop>=0:
        u1 = t.partition(stop)
    if qmax>0:
        u2 = t.split(qmax)

    if u1.max()<u2.max():
        u = u2
    else:
        u = u1

    cost = t.get_height()
    cost = cost[t.isleaf()==False]
    
    return u,cost


def Ward_segment(G,feature,stop=-1,qmax=1,verbose = 0):
    """
    Agglomerative function based on a topology-defining graph
    and a feature matrix. 
    INPUT:
    - G the input graph (a topological graph essentially)
    - feature (G.V,dim_feature) array that yields some vectorial information related to the graph vertices
    - stop = -1: the stopping crterion
    if stop==-1, then no stopping criterion is used
    - qmax=1: the maximum number of desired clusters
    (in the limit of the stopping criterion)
    OUPUT:
    -u: a labelling of the graph vertices according to the criterion
    -cost the cost of each merge step during the clustering procedure
    NOTE:
    - A euclidean distance is used in the feature space
    - caveat : when the number of cc in G (nbcc)
    is greter than qmax, u contains nbcc values, not qmax !
    """
    # basic check
    if np.size(feature)==feature.shape[0]:
        feature = np.reshape(feature, np.size(feature),1)

    if feature.shape[0]!=G.V:
        raise ValueError, "Incompatible dimension for the feature matrix and the graph"
    
    # prepare a graph with twice the number of vertices
    n = G.V
    if qmax==-1: qmax = n-1
    if stop==-1: stop = np.infty
    qmax = int(np.minimum(qmax,n-1))

    t = Ward(G,feature,verbose)
    if verbose: t.plot()

    u1 = np.zeros(n, np.int)
    u2 = np.zeros(n, np.int)
    if stop>=0:
        u1 = t.partition(stop)
    if qmax>0:
        u2 = t.split(qmax)

    if u1.max()<u2.max():
        u = u2
    else:
        u = u1

    cost = t.get_height()
    cost = cost[t.isleaf()==False]
    return u,cost


def Ward_simple(feature, verbose=0, quick=1):
    """
    Ward clustering based on a Feature matrix

    t = Ward(feature,verbose=0):
    INPUT:
    - feature: a (n,p) feature matrix between some items
    representing n p-dimenional items to be clustered
    - verbose=0, verbosity level
    OUTPUT:
    -t a weightForest structure that represents the dendrogram of the data
    NOTE:
    this method uses the optimized C routine if "quick" is true.
    """
    n = feature.shape[0]
    q = 2*n-1 
    if feature.ndim==1:
        feature = np.reshape(feature, (-1, 1))

    if quick:
        parent,height = ward(feature)
        t = WeightedForest(q,parent,height)
        return t
    
    # build Features as a list
    Features = [np.reshape(f, (1, -1)) for f in feature]
    #Features = []
    #for i in range(n):
    #   Features.append(np.reshape(feature[i],(1,feature.shape[1])))

    # create a distance matrix
    D = np.infty*np.ones((q,q))
    for i in range(n):
        for j in range(i):
            ESS = _inertia(i,j,Features)
            D[i,j] = ESS

    # prepare the main fields
    parent = np.arange(q).astype(np.int)
    height = np.zeros(q)

    # recursive merge loop
    for k in range(n,q):
        # identify the merge
        d = D.min()
        height[k] = d   
        i,j = np.where(D==d)
        i = i[0]
        j = j[0]
        parent[i] = k
        parent[j] = k

        # update the Features
        totalFeatures = np.vstack((Features[i],Features[j]))
        Features.append(totalFeatures)
        Features[i] = []
        Features[j] = []

        # update the distance
        for l in range(k):
            if parent[l]==l:
                D[k,l] =  _inertia(l,k,Features)
        D[i,:] = np.infty
        D[j,:] = np.infty
        D[:,i] = np.infty
        D[:,j] = np.infty

    # crate the resulting tree
    t = WeightedForest(q,parent,height)

    return t


def Ward(G, feature, verbose=0):
    """
    Agglomerative function based on a topology-defining graph
    and a feature matrix. 
    INPUT:
    - G the input graph (a topological graph essentially)
    - feature (G.V,dim_feature) array that yields some vectorial information related to the graph vertices
    OUPUT:
    -t: a WeightedForest structure that represents the dendrogram
    NOTE:
    When G has more than 1 connected component, t is no longer a tree.
    This case is handled cleanly now
    """
    # basic check
    if np.size(feature)==feature.shape[0]:
        feature = np.reshape(feature, np.size(feature),1)

    if feature.shape[0]!=G.V:
        raise ValueError, "Incompatible dimension for the feature matrix and the graph"
    
    Features = []
    for i in range(G.V):
        Features.append(np.reshape(feature[i],(1,feature.shape[1])))
    
    # prepare a graph with twice the number of vertices
    # this graph will contain the connectivity information
    # along the merges.
    n = G.V
    nbcc = G.cc().max()+1
    K = _auxiliary_graph(G,Features)

    # prepare some variables that are useful tp speed up the algorithm 
    parent = np.arange(2*n-nbcc).astype(np.int)
    pop = np.ones(2*n-nbcc).astype(np.int)
    height = np.zeros(2*n-nbcc)
    linc = K.left_incidence()
    rinc = K.right_incidence()

    # iteratively merge clusters
    for q in range(n-nbcc):
        # 1. find the lightest edge
        m = (K.weights).argmin()
        cost = K.weights[m]
        k = q+n
        i = K.edges[m,0]
        j = K.edges[m,1]
        height[k] = cost
        if verbose:
            print q,i,j, m,cost
        
        # 2. remove the current edge
        K.edges[m,:] = -1
        K.weights[m] = np.infty
        linc[i].remove(m)
        rinc[j].remove(m)
        
        ml = linc[j]
        if np.sum(K.edges[ml,1]==i)>0:
            m = ml[np.flatnonzero(K.edges[ml,1]==i)]
            K.edges[m,:] = -1
            K.weights[m] = np.infty
            linc[j].remove(m)
            rinc[i].remove(m)
        
        # 3. merge the edges with third part edges
        parent[i] = k
        parent[j] = k
        totalFeatures = np.vstack((Features[i],Features[j]))
        Features.append(totalFeatures)
        Features[i] = []
        Features[j] = []
        linc,rinc = _remap(K,i,j,k,Features,linc,rinc)

    # build a tree to encode the results
    t = WeightedForest(2*n-nbcc,parent,height)
    return t



#--------------------------------------------------------------------------
#------------- Maximum link clustering ------------------------------------
# -------------------------------------------------------------------------

def Maximum_Link_Euclidian(X,verbose=0):
    """
    Maximum link clustering based on data matrix.
    INPUT:
    - X: a (nbitem,dim) array from which an Euclidian distance
    matrix is computed
    - verbose=0, verbosity level
    OUTPUT:
    -t a weightForest structure that represents the dendrogram of the data
    NOTE:
    this method has not been optimized
    """
    if X.shape[0]==np.size(X):
        X = np.reshape(X,(np.size(X),1))
    if np.size(X)<10000:
        D = Euclidian_distance(X)
    else:
        raise ValueError, "The distance matrix is too large"
    
    t = Maximum_Link_Distance(D,verbose)
    return t 


def Maximum_Link_Distance(D,stop=-1,qmax=-1,verbose=0):
    """
    maximum link clustering based on a pairwise distance matrix.
    Maximum_Link_Distance(D,stop=0,qmax=-1,verbose=0):
    INPUT:
    - D: a (n,n) distance matrix between some items
    - verbose=0, verbosity level
    OUTPUT:
    -t a weightForest structure that represents the dendrogram of the data
    NOTE:
    this method has not been optimized
    """
    n = D.shape[0]
    if D.shape[1]!=n:
        raise ValueError, "non -square distance matrix" 
    if stop==-1:
        stop = np.infty

    DI = np.infty*np.ones((2*n,2*n))
    DI[:n,:n]=D+np.diag(np.infty*np.ones(n))
    parent = np.arange(2*n-1, dtype=np.int)
    pop = np.ones(2*n-1, np.int)
    height = np.zeros(2*n-1)

    for q in range(n-1):
        d = DI.min()
        k = q+n
        height[k] = d
        
        i,j = np.where(DI==d)
        i = i[0]
        j = j[0]
        parent[i] = k
        parent[j] = k
        DI[k] = np.maximum(DI[i],DI[j])
        DI[:,k] = np.transpose(DI[k,:])
        DI[i,:] = np.infty
        DI[j,:] = np.infty
        DI[:,i] = np.infty
        DI[:,j] = np.infty

    t = WeightedForest(2*n-1,parent,height)

    return t

def Maximum_Link_Distance_segment(D,stop=-1,qmax=1,verbose=0):
    """
    maximum link clustering based on a pairwise distance matrix.
    Maximum_Link_Distance(D,stop=0,qmax=-1,verbose=0):
    INPUT:
    - D: a (n,n) distance matrix between some items
    - qmax = 1: the number of desired clusters
    (in the limit of stop)
    - stop=-1: stopping criterion, i.e. distance threshold at which
    further merges are forbidden
    By default (stop=-1), all merges are performed
    - verbose=0, verbosity level
    OUTPUT:
    -u: a labelling of the graph vertices according to the criterion
    -cost the cost of each merge step during the clustering procedure
    NOTE:
    this method has not been optimized
    """
    n = D.shape[0]
    if D.shape[1]!=n:
        raise ValueError, "non -square distance matrix" 
    if stop==-1:
        stop = np.infty

    t = Maximum_Link_Distance(D,verbose)
    if verbose: t.plot()

    u1 = np.zeros(n, np.int)
    u2 = np.zeros(n, np.int)
    if stop>=0:
        u1 = t.partition(stop)
    if qmax>0:
        u2 = t.split(qmax)

    if u1.max()<u2.max():
        u = u2
    else:
        u = u1

    cost = t.get_height()
    cost = cost[t.isleaf()==False]

    return u,cost

#--------------------------------------------------------------------------
#----------------------- Visualization ------------------------------------
# -------------------------------------------------------------------------

def _label_(f,parent,left,labelled):
    temp = np.nonzero(parent==f)

    if np.size(temp)>0:
        i = temp[0][np.nonzero(left[temp[0]]==1)]
        j = temp[0][np.nonzero(left[temp[0]]==0)]
        labelled = _label_(i,parent,left,labelled)
        #print i,j,f
        labelled[f] = labelled.max()+1
        labelled = _label_(j,parent,left,labelled)

    if labelled[f]<0:   
        labelled[f] = labelled.max()+1

    return labelled

        
def _label(parent):
    # find the root
    root  = np.nonzero(parent == np.arange(np.size(parent)))[0]
    
    # define left
    left = np.zeros(np.size(parent))
    for f in range(np.size(parent)):
        temp = np.nonzero(parent==f)
        if np.size(temp)>0:
            left[temp[0][0]] = 1

    left[root]=.5

    # define labelled
    labelled = -np.ones(np.size(parent))
    
    # compute labelled
    for j in range(np.size(root)):
        labelled = _label_(root[j],parent,left,labelled)


    return labelled


def _label2(parent, children):
    # find the root
    root  = np.nonzero(parent == np.arange(np.size(parent)))[0]

    # define left
    left = np.zeros(np.size(parent))
    for ch in children:
        if np.size(ch)>1:
            left[ch[0]] = 1
    
    left[root]=.5

    # define labelled
    labelled = -np.ones(np.size(parent))

    # compute labelled
    for j in range(np.size(root)):
        labelled = _label_2(root[j],parent,left,labelled)
    
    return labelled

def _label_2(f,parent,left,labelled):
    temp = np.nonzero(parent==f)
    if np.size(temp)>0:
        i = temp[0][np.nonzero(left[temp[0]]==1)]
        j = temp[0][np.nonzero(left[temp[0]]==0)]
        labelled = _label_2(i,parent,left,labelled)
        #print i,j,f
        labelled[f] = labelled.max()+1
        for jj in j:
            labelled = _label_2(jj,parent,left,labelled)
    if labelled[f]<0:
        labelled[f] = labelled.max()+1
    return labelled




