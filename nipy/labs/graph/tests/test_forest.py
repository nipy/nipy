import numpy as np
from ..forest import  Forest

def simple_forest():
    """ generate a simple forest
    """
    parents = np.array([2, 2, 4, 4, 4])
    return Forest(5, parents)

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
    

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

