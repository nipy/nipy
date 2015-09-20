from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from ..forest import  Forest

def simple_forest():
    """ generate a simple forest
    """
    parents = np.array([2, 2, 4, 4, 4])
    F = Forest(5, parents)
    return F

def test_forest():
    """ test creation of forest object
    """
    F = simple_forest()
    assert F.E == 8
    assert F.cc().max() == 0

def test_forest_trivial():
    """ test creation of forest object
    """
    F = Forest(5)
    assert F.E == 0
    assert (F.cc() == np.arange(5)).all()

def test_children():
    """ test that we obtain children
    """
    sf = simple_forest()
    ch = sf.get_children()
    assert len(ch) == 5
    assert ch[0] == []
    assert ch[1] == []
    assert ch[2] == [0, 1]
    assert ch[3] == []
    assert ch[4] == [2, 3]
    
def test_descendants():
    """ test the get_descendants() method
    """
    sf = simple_forest()
    assert sf.get_descendants(0) == [0]
    assert sf.get_descendants(1) == [1]
    assert sf.get_descendants(2) == [0, 1, 2]
    assert sf.get_descendants(4) == [0, 1, 2, 3, 4]
    
def test_root():
    """ test the isroot() method
    """ 
    root = simple_forest().isroot()
    assert root[4] == True
    assert root.sum() == 1

def test_merge_simple_branches():
    """ test the merge_simple_branches() method
    """
    f = Forest(5, np.array([2, 2, 4, 4, 4])).merge_simple_branches()
    assert f.V == 5
    f = Forest(5, np.array([1, 2, 4, 4, 4])).merge_simple_branches()
    assert f.V == 3

def test_all_distances():
    """ test the all_distances() methods
    """
    f =  simple_forest()
    dg = f.all_distances()
    print(dg)
    assert dg[0, 3] == 3.
    assert dg.max() == 3.
    assert dg.min() == 0.
    assert dg.shape == (5, 5)
    dg = f.all_distances(1)
    assert dg[3] == 3.

def test_depth():
    """ test the depth_from_leaves() methods
    """
    f =  simple_forest()
    depth = f.depth_from_leaves()
    assert depth[0] == 0
    assert depth[1] == 0
    assert depth[3] == 0
    assert depth[2] == 1
    assert depth[4] == 2
    
def test_reorder():
    """ test the reorder_from_leaves_to_roots() method
    """
    f =  simple_forest()
    order = f.reorder_from_leaves_to_roots()
    assert (f.depth_from_leaves() == np.array([0, 0, 0, 1, 2])).all()
    assert (order == np.array([0, 1, 3, 2, 4])).all()

def test_leaves():
    """ test the leaves_of_a_subtree() method
    """
    f =  simple_forest()
    assert f.leaves_of_a_subtree([0, 1]) == True
    assert f.leaves_of_a_subtree([0, 3]) == False
    assert f.leaves_of_a_subtree([1, 3]) == False
    assert f.leaves_of_a_subtree([0, 1, 3]) == True
    assert f.leaves_of_a_subtree([1]) == True
   
def test_depth():
    """ Test the tree_depth() method
    """
    f =  simple_forest()
    assert f.tree_depth() == 3

def test_upward_and():
    """ test the propagate_upward_and() method
    """
    f =  simple_forest()
    assert(f.propagate_upward_and([0, 1, 0, 1, 0]) == [0, 1, 0, 1, 0]).all()
    assert(f.propagate_upward_and([0, 1, 1, 1, 0]) == [0, 1, 0, 1, 0]).all()
    assert(f.propagate_upward_and([0, 1, 1, 1, 1]) == [0, 1, 0, 1, 0]).all()
    assert(f.propagate_upward_and([1, 1, 0, 1, 0]) == [1, 1, 1, 1, 1]).all()

def test_upward():
    """ test the propagate_upward() method
    """
    f =  simple_forest()
    assert(f.propagate_upward([0, 0, 1, 3, 1]) == [0, 0, 0, 3, 1]).all()
    assert(f.propagate_upward([0, 0, 5, 0, 2]) == [0, 0, 0, 0, 0]).all()



if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

