# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Demo ward clustering on a graph:
various ways of forming clusters and dendrogram
"""
print __doc__

import numpy as np
from numpy.random import randn, rand
import matplotlib.pylab as mp

from nipy.labs import graph
from nipy.labs.clustering.hierarchical_clustering import ward

# n = number of points, k = number of nearest neighbours
n = 100
k = 5
verbose = 0

X = randn(n, 2)
X[:np.ceil(n / 3)] += 3		
G = graph.WeightedGraph(n)
G.knn(X, 5)
tree = ward(G, X, verbose)

threshold = .5 * n
u = tree.partition(threshold)

mp.figure()
mp.subplot(1, 2, 1)
for i in range(u.max()+1):
    mp.plot(X[u == i, 0], X[u == i, 1], 'o', color=(rand(), rand(), rand()))

mp.axis('tight')
mp.axis('off')
mp.title('clustering into clusters of inertia<%f' % threshold)

u = tree.split(k)
mp.subplot(1, 2, 2)
for e in range(G.E):
    mp.plot([X[G.edges[e, 0], 0], X[G.edges[e, 1], 0]],
            [X[G.edges[e, 0], 1], X[G.edges[e, 1], 1]], 'k')
for i in range(u.max() + 1):
    mp.plot(X[u == i, 0], X[u == i, 1], 'o', color=(rand(), rand(), rand()))
mp.axis('tight')
mp.axis('off')
mp.title('clustering into 5 clusters')

nl = np.sum(tree.isleaf())
validleaves = np.zeros(n)
validleaves[:np.ceil(n / 4)] = 1
valid = np.zeros(tree.V, 'bool')
valid[tree.isleaf()] = validleaves.astype('bool')
nv = np.sum(validleaves)
nv0 = 0
while nv > nv0:
    nv0 = nv
    for v in range(tree.V):
        if valid[v]:
            valid[tree.parents[v]]=1
        nv = np.sum(valid)

ax = tree.plot()
ax.set_visible(True)
mp.show()

if verbose:
    print 'List of sub trees'
    print tree.list_of_subtrees()
