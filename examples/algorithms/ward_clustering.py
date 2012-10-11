#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function # Python 2/3 compatibility
__doc__ = """
Demo ward clustering on a graph: various ways of forming clusters and dendrogram

Requires matplotlib
"""
print(__doc__)

import numpy as np
from numpy.random import randn, rand

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nipy.algorithms.graph import knn
from nipy.algorithms.clustering.hierarchical_clustering import ward

# n = number of points, k = number of nearest neighbours
n = 100
k = 5

# Set verbose to True to see more printed output
verbose = False

X = randn(n, 2)
X[:np.ceil(n / 3)] += 3
G = knn(X, 5)
tree = ward(G, X, verbose)

threshold = .5 * n
u = tree.partition(threshold)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
for i in range(u.max()+1):
    plt.plot(X[u == i, 0], X[u == i, 1], 'o', color=(rand(), rand(), rand()))

plt.axis('tight')
plt.axis('off')
plt.title('clustering into clusters \n of inertia < %g' % threshold)

u = tree.split(k)
plt.subplot(1, 3, 2)
for e in range(G.E):
    plt.plot([X[G.edges[e, 0], 0], X[G.edges[e, 1], 0]],
            [X[G.edges[e, 0], 1], X[G.edges[e, 1], 1]], 'k')
for i in range(u.max() + 1):
    plt.plot(X[u == i, 0], X[u == i, 1], 'o', color=(rand(), rand(), rand()))
plt.axis('tight')
plt.axis('off')
plt.title('clustering into 5 clusters')

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

ax = plt.subplot(1, 3, 3)
ax = tree.plot(ax)
ax.set_title('Dendrogram')
ax.set_visible(True)
plt.show()

if verbose:
    print('List of sub trees')
    print(tree.list_of_subtrees())
