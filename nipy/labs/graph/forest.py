# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This module implements the DAG class of Forest(WeightedGraph):
This is a special case of a Weighted Graph (i.e.  a set of trees)

Main author: Bertrand thirion, 2007-2011
"""

import numpy as np
from graph import WeightedGraph, graph_to_neighb


class Forest(WeightedGraph):
    """
    This is a Forest structure, i.e. a set of trees
    The nodes can be segmented into trees
    Within each tree a node has one parent and children
        (hierarchical structure)
    Some of the nodes can be viewed as leaves, other as roots
    The edges within a tree are associated with a weight:
    +1 from child to parent
    -1 from parent to child

    fields
    ------
    V : (int,>0) the number of vertices
    E : (int) the number of edges
    parents: array of shape (self.V) the parent array
    edges: array of shape (self.E,2) reprensenting pairwise neighbors
    weights, array of shape (self.E), +1/-1 for scending/descending links
    children: list of arrays that represents the childs of any node
    """

    def __init__(self, V, parents=None):
        """Constructor

        Parameters
        ----------
        V (int), the number of edges of the graph
        parents = None: array of shape (V)
                the parents of zach vertex
                if Parents==None , the parents are set to range(V), i.e. each
                node is its own parent, and each node is a tree
        """
        V = int(V)
        if V < 1:
            raise ValueError('cannot create graphs with no vertex')
        self.V = int(V)

        # define the parents
        if parents == None:
            self.parents = np.arange(self.V).astype(np.int)
        else:
            if np.size(parents) != V:
                raise ValueError('Incorrect size for parents')
            if parents.max() > self.V:
                raise ValueError('Incorrect value for parents')

            self.parents = np.reshape(parents, self.V).astype(np.int)

        self.define_graph_attributes()

        if self.check() == 0:
            raise ValueError('The proposed structure is not a forest')
        self.children = []

    def define_graph_attributes(self):
        """
        define the edge and weights array
        """
        self.edges = np.array([]).astype(np.int)
        self.weights = np.array([])
        i = np.nonzero(self.parents != np.arange(self.V))[0]
        if np.size(i) > 0:
            E1 = np.hstack((i, self.parents[i]))
            E2 = np.hstack((self.parents[i], i))
            self.edges = (np.vstack((E1, E2))).astype(np.int).T
            self.weights = np.hstack((np.ones(np.size(i)),
                                      - np.ones(np.size(i))))

        self.E = np.size(self.weights)
        self.edges = self.edges

    def compute_children(self):
        """
        self.compute_children()
        define the children list

        Returns
        -------
        children: a list of self.V lists,
                  that yields the children of each node
        """
        self.children = [np.array([]) for v in range(self.V)]
        if self.E > 0:
            i = self.weights < 0
            a = self.edges[i, 0]
            b = self.edges[i, 1]
            ci, ne, we = graph_to_neighb(a, b, np.ones(np.size(a)), self.V)

            self.children = [np.array(ne[ci[v]:ci[v + 1]])
                              for v in range(self.V)]

    def get_children(self, v=-1):
        """
        returns the list list of children arrays in all the forest
        if v==-1 or the children of v otherwise
        """
        v = int(v)
        if v > -1:
            if v > self.V-1:
                raise ValueError('the given node index is too high')
        if self.children == []:
            self.compute_children()
        if v == -1:
            return self.children
        else:
            return self.children[v]

    def get_descendents(self, v):
        """
        returns the nodes that are children of v
        """
        v = int(v)
        if v < 0:
            raise ValueError('the given node index is too low')
        if v > self.V - 1:
            raise ValueError('the given node index is too high')
        if self.children == []:
            self.compute_children()
        if len(self.children[v]) == 0:
            return [v]
        else:
            desc = [v]
            for w in self.children[v]:
                temp = self.get_descendents(w)
                for q in temp:
                    desc.append(q)
        return desc

    def check(self):
        """
        Check that the proposed  is indeed a graph, i.e. contains no loop

        Returns
        -------
        a boolean b=0 iff there are loops, 1  otherwise

        Note
        ----
        slow implementation, might be rewritten in C or cython
        """
        b = 1
        if self.V == 1:
            return b
        for v in range(self.V):
            w = v
            q = 0
            while(self.parents[w] != w):
                w = self.parents[w]
                if w == v:
                    b = 0
                    break
                q += 1
                if q > self.V:
                    b = 0
                    break
            if b == 0:
                break
        return b

    def isleaf(self):
        """
        returns a bool array of shape(self.V)
        so that isleaf==1 iff the node is a leaf in the forest
        (has no kids)
        """
        leaves = np.ones(self.V).astype('bool')
        if self.E > 0:
            leaves[self.edges[self.weights > 0, 1]] = 0
        return leaves

    def isroot(self):
        """
        returns a bool array of shape(self.V)
        so that isleaf==1 iff the node is a root in the forest
        i.e. : is its own parent
        """
        roots = np.array(self.parents == np.arange(self.V))
        return roots

    def subforest(self, valid):
        """
        creates a subforest with the vertices
        for which valid>0
        and with the correponding set of edges
        the children of deleted vertices become
        their own parent

        Parameters
        ----------
        valid: array of shape (self.V)

        Returns
        -------
        a new forest instance
        """
        if np.size(valid) != self.V:
            raise ValueError("incompatible size for self anf valid")

        parents = self.parents.copy()
        j = np.nonzero(valid[self.parents] == 0)[0]
        parents[j] = j
        parents = parents[valid]
        renumb = np.hstack((0, np.cumsum(valid)))
        parents = renumb[parents]

        F = Forest(np.sum(valid), parents)

        return F

    def merge_simple_branches(self):
        """
        merge the branches of the forest that are the only child
        of the parent branch into their child
        """
        valid = np.ones(self.V).astype('bool')
        children = self.get_children()
        for k in range(self.V):
            if np.size(children[k]) == 1:
                valid[k] = 0
        return self.subforest(valid)

    def all_distances(self, seed=None):
        """
        returns all the distances of the graph  as a tree

        Parameters
        ----------
        seed=None array of shape(nbseed)  with valuesin [0..self.V-1]
                  set of vertices from which tehe distances are computed

        Returns
        -------
        dg: array of shape(nseed, self.V), the resulting distances

        Note
        ----
        by convention infinte distances are given the distance np.infty
        """
        if self.E > 0:
            w = self.weights.copy()
            self.weights = np.absolute(self.weights)
            dg = self.floyd(seed)
            dg[dg == (np.sum(self.weights) + 1)] = np.infty
            self.weights = w
            return dg
        else:
            return np.array([])

    def depth_from_leaves(self):
        """
        compute a labelling of the nodes
        which is 0 for the leaves, 1 for their parents etc
        and maximal for the roots

        Returns
        -------
        depth: array of shape (self.V): the depth values of the vertices
        """
        depth = self.isleaf().astype(np.int)-1
        for j in range(self.V):
            dc = depth.copy()
            for i in range(self.V):
                if self.parents[i] != i:
                    depth[self.parents[i]] = np.maximum(depth[i] + 1,\
                                           depth[self.parents[i]])
            if dc.max() == depth.max():
                break
        return depth

    def reorder_from_leaves_to_roots(self):
        """
        reorder the tree so that the leaves come first
        then their parents and so on, and the roots are last
        the permutation necessary to apply to all vertex-based information

        Returns
        -------
        order: array of shape(self.V)
               the order of the old vertices in the reordered graph
        """
        depth = self.depth_from_leaves()
        order = np.argsort(depth)
        iorder = np.arange(self.V)
        for i in range(self.V):
            iorder[order[i]] = i
        parents = iorder[self.parents[order]]
        self.parents = parents
        self.define_graph_attributes()
        return order

    def leaves_of_a_subtree(self, ids, custom=False):
        """
        tests whether the given nodes within ids represent
        all the leaves of a certain subtree of self

        Parameters
        ----------
        idds: array of shape (n) that takes values in [0..self.V-1]
        custom == False, boolean
               if custom==true the behavior of the function is more specific
               - the different connected components are considered
               as being in a same greater tree
               - when a node has more than two subbranches,
               any subset of these children is considered as a subtree
        """
        leaves = self.isleaf().astype('bool')
        for i in ids:
            if leaves[i] == 0:
                raise ValueError("some of the ids are not leaves")

        #1. find the highest node that is a common ancestor to all leaves
        # if there is none, common ancestor is -1
        com_ancestor = ids[0]
        for i in ids:
            ca = i
            dca = self.get_descendents(ca)
            while com_ancestor not in dca:
                ca = self.parents[ca]
                dca = self.get_descendents(ca)
                if (ca == self.parents[ca]) & (com_ancestor not in dca):
                    ca = -1
                    break
            com_ancestor = ca

        #2. check whether all the children of this ancestor are within ids
        if com_ancestor > -1:
            st = self.get_descendents(com_ancestor)
            valid = [i in ids for i in st if leaves[i]]
            bresult = (np.sum(valid) == np.size(valid))
            if custom == False:
                return bresult

            # now, custom =True
            # check that subtrees of ancestor are consistently labelled
            kids = self.get_children(com_ancestor)
            if np.size(kids) > 2:
                bresult = True
                for v in kids:
                    st = np.array(self.get_descendents(v))
                    st = st[leaves[st]]
                    if np.size(st) > 1:
                        valid = [i in ids for i in st]
                        bresult *= ((np.sum(valid) == np.size(valid))
                                    + np.sum(valid == 0))
            return bresult

        # now, common ancestor is -1
        if custom == False:
            st = np.squeeze(np.nonzero(leaves))
            valid = [i in ids for i in st]
            bresult = (np.sum(valid) == np.size(valid))
        else:
            cc = self.cc()
            bresult = True
            for i in ids:
                st = np.squeeze(np.nonzero((cc == cc[i]) * leaves))
                if np.size(st) > 1:
                    valid = [i in ids for i in st]
                    bresult *= (np.sum(valid) == np.size(valid))
                else:
                    bresult *= (st in ids)
        return bresult

    def tree_depth(self):
        """
        return the maximal depth of any node in the tree
        """
        depth = self.depth_from_leaves()
        return depth.max() + 1

    def propagate_upward_and(self, prop):
        """
        propagates some binary property in the forest
        that is defined in the leaves
        so that prop[parents] = logical_and(prop[children])

        Parameters
        ----------
        prop, array of shape(self.V), the input property

        Returns
        -------
        prop, array of shape(self.V), the output property field
        """
        if np.size(prop) != self.V:
            raise ValueError("incoherent size for prop")

        prop[self.isleaf() == False] = True

        for j in range(self.tree_depth()):
            for i in range(self.V):
                if prop[i] == False:
                    prop[self.parents[i]] = False

        return prop

    def propagate_upward(self, label):
        """
        label = self.propagate_upward(label)
        Assuming that label is a certain positive integer field
        (i.e. labels)
        that is defined at the leaves of the tree
        and can be compared,
        this propagates these labels to the parents whenever
        the children nodes have coherent properties
        otherwise the parent value is unchanged

        Parameters
        ----------
        label: array of shape(self.V)

        Returns
        -------
        label: array of shape(self.V)
        """
        if np.size(label) != self.V:
            raise ValueError("incoherent size for label")

        ch = self.get_children()
        depth = self.depth_from_leaves()
        for j in range(1, depth.max() + 1):
            for i in range(self.V):
                if depth[i] == j:
                    if np.size(np.unique(label[ch[i]])) == 1:
                        label[i] = np.unique(label[ch[i]])
        return label

    def rooted_subtree(self, k):
        """
        l = self.subtree(k)
        returns an array of the nodes included in the subtree rooted in k

        Parameters
        ----------
        k (int): the vertex from which the subtree is searched

        Returns
        -------
        idx: array of shape>=1, the index of the nodes beneath k

        Fixme
        -----
        should return a Forest and/or a vector of booleans
        """
        if ((k > self.V) or (k < 0)):
            raise ValueError("incoherent value for k")

        valid = np.zeros(self.V)
        valid[k] = 1
        sk = 0
        while valid.sum() > sk:
            sk = valid.sum().copy()
            for i in range(self.V):
                if valid[self.parents[i]] == 1:
                    valid[i] = 1

        idx = np.nonzero(valid)
        idx = np.reshape(idx, np.size(idx))
        return idx
