import numpy.random as R

maxtag = 10**8
def new():
    """
    Generate a tag -- obviously there is a better way.
    """
    
    return R.random_integers(0, maxtag)
