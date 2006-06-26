#-----------------------------------------------------------------------------
#
#  Copyright (c) 2005, Enthought, Inc.
#  All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.neuroimaging.extra.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: Enthought, Inc.
# Description: <Enthought util package component>
#
#-----------------------------------------------------------------------------

"""A collection of graph algorithms that work on graphs defined by a
dictionary which is an adjacency list.  node maps to a list of successor
nodes.
"""

class CyclicGraph(Exception):
    """
    Exception for cyclic graphs.
    """
    def __init__(self):
        Exception.__init__(self, "Graph is cyclic")


def topological_sort(graph):
    """
    Returns the nodes in the graph in topological order.
    """
    discovered = {}
    explored = {}
    order = []
    def explore(node):
        children = graph.get(node, [])
        for child in children:
            if child in explored:
                pass
            elif child in discovered:
                raise CyclicGraph()
            else:
                discovered[child] = 1
                explore(child)
        explored[node] = 1
        order.append(node)
    
    for node in graph.keys():
        if node not in explored:
            explore(node)
    order.reverse()
    return order
                

def closure(graph, sorted=True):
    """
    Returns the transitive closure of the graph.
    If sorted is True then the successor nodes will
    be sorted into topological order.
    """
    order = topological_sort(graph)
    reachable = {}
    for i in range(len(order)-1, -1, -1):
        node = order[i]
        # We are going through in reverse topological order so we
        # are guaranteed that all of the children of the node
        # are already in reachable
        node_reachable = {}
        for child in graph.get(node, []):
            node_reachable[child] = 1
            node_reachable.update(reachable[child])
        reachable[node] = node_reachable
    # Now, build the return graph by doing a topological sort of
    # each reachable set, if required
    retval = {}
    indexes = {}
    def node_index(node):
        i = indexes.get(node)
        if i is None:
            i = order.index(node)
        return i
    def cmpfunc(node1, node2):
        return cmp(node_index(node1), node_index(node2))
    for node, node_reachable in reachable.items():
        if not sorted:
            retval[node] = node_reachable.keys()
        else:
            reachable_list = node_reachable.keys()[:]
            reachable_list.sort(cmpfunc)
            retval[node] = reachable_list
    return retval

def reverse(graph):
    """
    Returns the reverse of a graph, that is the graph made when all
    of the edges are reversed.
    """
    retval = {}
    for node, successors in graph.items():
        for s in successors:
            retval.setdefault(s, []).append(node)
    return retval

if __name__ == "__main__":
    g = {1:[2,3],
         2:[3,4],
         6:[3],
         4:[6]}
    print topological_sort(g)
    print closure(g)

#### EOF ######################################################################
