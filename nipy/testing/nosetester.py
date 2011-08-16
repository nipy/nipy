""" Nipy nosetester

Sets doctests to run by default
Use our own doctest plugin (based on that of numpy)
"""

from ..fixes.numpy.testing.nosetester import NoseTester, import_nose

class NipyNoseTester(NoseTester):
    """ Numpy-like testing class

    * Removes some numpy-specific excludes
    * Disables numpy's fierce clearout of module import context for doctests
    * Run doctests by default
    """
    excludes = []

    def _get_custom_doctester(self):
        """ Use our our own doctester """
        import_nose()
        from .doctester import NipyDoctest
        return NipyDoctest()

    def test(self, label='fast', verbose=1, extra_argv=None, doctests=True,
             coverage=False):
        """
        Run tests for module using nose.

        As for numpy tester, except enable tests by default.

        Parameters
        ----------
        label : {'fast', 'full', '', attribute identifier}, optional
            Identifies the tests to run. This can be a string to pass to
            the nosetests executable with the '-A' option, or one of several
            special values.  Special values are:
            * 'fast' - the default - which corresponds to the ``nosetests -A``
              option of 'not slow'.
            * 'full' - fast (as above) and slow tests as in the
              'no -A' option to nosetests - this is the same as ''.
            * None or '' - run all tests.
            attribute_identifier - string passed directly to nosetests as '-A'.
        verbose : int, optional
            Verbosity value for test outputs, in the range 1-10. Default is 1.
        extra_argv : list, optional
            List with any extra arguments to pass to nosetests.
        doctests : bool, optional
            If True, run doctests in module. Default is True.
        coverage : bool, optional
            If True, report coverage of nipy code. Default is False.
            (This requires the `coverage module:
             <http://nedbatchelder.com/code/modules/coverage.html>`_).

        Returns
        -------
        result : object
            Returns the result of running the tests as a
            ``nose.result.TextTestResult`` object.

        Notes
        -----
        Each nipy module should expose `test` in its namespace to run all tests
        for it.  For example, to run all tests for nipy.algorithms:

        >>> import nipy.algorithms
        >>> nipy.algorithms.test() #doctest: +SKIP
        """
        return super(NipyNoseTester, self).test(label, verbose, extra_argv,
                                                doctests, coverage)
