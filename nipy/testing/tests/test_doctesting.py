""" Doctests for Nipy / NumPy-specific nose/doctest modifications
"""
# try the #random directive on the output line
def check_random_directive():
    '''
    >>> 2+2
    <BadExample object at 0x084D05AC>  #random: may vary on your system
    '''

# check the implicit "import numpy as np"
def check_implicit_np():
    '''
    >>> np.array([1,2,3])
    array([1, 2, 3])
    '''

# there's some extraneous whitespace around the correct responses
def check_whitespace_enabled():
    '''
    # whitespace after the 3
    >>> 1+2
    3

    # whitespace before the 7
    >>> 3+4
     7
    '''

def check_empty_output():
    """ Check that no output does not cause an error.

    This is related to nose bug 445; the numpy plugin changed the
    doctest-result-variable default and therefore hit this bug:
    http://code.google.com/p/python-nose/issues/detail?id=445

    >>> a = 10
    """

def check_skip():
    """ Check skip directive

    The test below should not run

    >>> 1/0 #doctest: +SKIP
    """

def func():
    return 1

def check_have_module_context():
    """ Check that, unlike numpy, we do have the module namespace

    >>> func()
    1
    """


if __name__ == '__main__':
    # Run tests outside nipy test rig
    import nose
    from nipy.testing.doctester import NipyDoctest
    argv = ['', __file__, '--with-nipydoctest']
    nose.core.TestProgram(argv=argv, addplugins=[NipyDoctest()])
