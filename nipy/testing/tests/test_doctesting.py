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

def check_fails():
    """ Check inversion directive

    The directive is mainly for tests

    >>> 'black' #doctest: +NOT_EQUAL
    'white'
    >>> 'white' #doctest: +NOT_EQUAL
    'black'
    """

def check_ignore_output():
    """ Check IGNORE_OUTPUT option works

    >>> 'The answer' #doctest: +IGNORE_OUTPUT
    42
    >>> 'The answer' #doctest: +IGNORE_OUTPUT
    'The answer'
    """

def check_sympy_equal():
    """ Check SYMPY_EQUAL option

    >>> from sympy import symbols
    >>> a, b, c = symbols('a, b, c')
    >>> a + b #doctest: +SYMPY_EQUAL
    b + a
    >>> a + b #doctest: +SYMPY_EQUAL
    a + b
    >>> a + b #doctest: +SYMPY_EQUAL +NOT_EQUAL
    a + c
    >>> a + b #doctest: +SYMPY_EQUAL +NOT_EQUAL
    a - b
    """

def check_fp_equal():
    """ Check floating point equal

    >>> 0.12345678 #doctest: +FP_6DP
    0.1234569
    >>> 0.12345678 #doctest: +FP_6DP +NOT_EQUAL
    0.1234564
    >>> 0.12345678 #doctest: +FP_4DP
    0.1235
    >>> 0.12345678 #doctest: +FP_6DP +NOT_EQUAL
    0.1235
    """

def check_array_repr():
    """ Stripping of array repr

    >>> arr = np.arange(5, dtype='i2')

    The test should match with and without the array repr

    >>> arr #doctest: +STRIP_ARRAY_REPR
    [0, 1, 2, 3, 4]
    >>> arr #doctest: +STRIP_ARRAY_REPR
    array([0, 1, 2, 3, 4], dtype=int16)
    """


def check_ignore_dtype():
    """ Stripping of dtype from array repr

    >>> arr = np.arange(5, dtype='i2')

    The test should match with and without the array repr

    >>> arr #doctest: +IGNORE_DTYPE
    array([0, 1, 2, 3, 4])
    >>> arr #doctest: +IGNORE_DTYPE
    array([0, 1, 2, 3, 4], dtype=int16)
    >>> arr #doctest: +IGNORE_DTYPE
    array([0, 1, 2, 3, 4], dtype=int32)
    >>> 1, arr, 3 #doctest: +IGNORE_DTYPE
    (1, array([0, 1, 2, 3, 4], dtype=int32), 3)
    """

def check_combinations():
    """ Check the processing combines as expected

    >>> 0.33333 #doctest: +SYMPY_EQUAL +NOT_EQUAL
    0.3333
    >>> 0.33333 #doctest: +SYMPY_EQUAL +FP_4DP
    0.3333
    >>> arr = np.arange(5, dtype='i2')

    This next will not sympify unless the array repr is removed

    >>> arr #doctest: +STRIP_ARRAY_REPR +SYMPY_EQUAL
    array([0, 1, 2, 3, 4], dtype=int16)
    """


if __name__ == '__main__':
    # Run tests outside nipy test rig
    import sys
    import nose
    from nipy.testing.doctester import NipyDoctest
    argv = [sys.argv[0], __file__, '--with-nipydoctest'] + sys.argv[1:]
    nose.core.TestProgram(argv=argv, addplugins=[NipyDoctest()])
