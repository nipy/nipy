''' Tests for contrasts module '''

from nipy.modalities.fmri.formula import Term, Formula

from nipy.modalities.fmri.contrast import Contrast

from nipy.testing import assert_true, assert_equal


def test_contrasts():
    x = Term('x')
    y = Term('y')
    f = Formula([x,y])
    con = Contrast(x, f, 'my contrast')
    yield assert_equal, con.term, x
    yield assert_equal, con.formula, f
    yield assert_equal, con.name, 'my contrast'
    # check at least that string does not crash
    s = str(con)
    # this raises an error; it looks like the contrast code is using the
    # old scipy stats models code
    m = con.matrix
