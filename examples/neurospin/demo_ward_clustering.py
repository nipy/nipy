import numpy as np
from numpy.random import randn, rand
import nipy.neurospin.graph as fg
import matplotlib.pylab as mp
from nipy.neurospin.clustering.hierarchical_clustering import ward

# n =number of points, k = number of nearest neighbours
n =100
k  = 5
verbose=0

X = randn(n,2)
X[:np.ceil(n/3)] += 3		
G = fg.WeightedGraph(n)
#G.mst(X)
G.knn(X,5)
t = ward(G,X,verbose)

print t.check_compatible_height()
ax = t.plot()
#t.plot_height()

u = t.partition(1.0)

mp.figure()
mp.subplot(1,2,1)
for i in range(u.max()+1):
    mp.plot(X[u==i,0],X[u==i,1],'o',color=(rand(),rand(),rand()))

mp.axis('tight')
mp.axis('off')
mp.title('clustering into clusters of inertia<1')

u = t.split(k)
mp.subplot(1,2,2)
for e in range(G.E):
    mp.plot([X[G.edges[e,0],0],X[G.edges[e,1],0]],
            [X[G.edges[e,0],1],X[G.edges[e,1],1]],'k')
for i in range(u.max()+1):
    mp.plot(X[u==i,0],X[u==i,1],'o',color=(rand(),rand(),rand()))
mp.axis('tight')
mp.axis('off')
mp.title('clustering into 5 clusters')



nl = np.sum(t.isleaf())
validleaves = np.zeros(n)
validleaves[:np.ceil(n/4)]=1
valid = np.zeros(t.V,'bool')
valid[t.isleaf()]=validleaves.astype('bool')
nv =  np.sum(validleaves)
nv0 = 0
while nv>nv0:
    nv0= nv
    for v in range(t.V):
        if valid[v]:
            valid[t.parents[v]]=1
    nv = np.sum(valid)
    
ax = t.fancy_plot_(valid)
ax.axis('off')

mp.show()

if verbose:
    print t.list_of_subtrees()
