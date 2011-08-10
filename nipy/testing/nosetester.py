""" Nipy nosetester

Removes some specialisms from the numpy NoseTester class

The main purpose is to use the usual doctest running plugin for nose rather than
the specialized numpy doctest plugin.
"""
import os

from numpy.testing.nosetester import NoseTester, import_nose

try:
    import nose.core
except ImportError:
    class NipyTestProgram(object):
        def __init__(self, *args, **kwargs):
            raise ImportError('Need nose for tests')
else:
    from numpy.testing.noseclasses import NumpyTestProgram
    class NipyTestProgram(NumpyTestProgram):
        # Disable rerouting of config by missing out override of makeConfig

        def makeConfig(self, *args, **kwargs):
            """Load a Config, pre-filled with user config files if any are
            found.

            We override this method only to prevent the NumpyTestProgram
            overriding it, because that pulls out the normal doctest plugin
            """
            return nose.core.TestProgram.makeConfig(self, *args, **kwargs)


class Unplugger(object):
    """ Nose plugin to remove named plugin late in loading

    By default it removes the "doctest" plugin.

    Note the code does not use nose, so we don't have to shield it from import
    when we don't have nose.
    """
    name = 'unplugger'
    enabled = True # always enabled
    score = 4000 # load late in order to be after builtins

    def __init__(self, to_unplug='doctest'):
        self.to_unplug = to_unplug

    def options(self, parser, env):
        pass

    def configure(self, options, config):
        # Pull named plugin out of plugins list
        config.plugins.plugins = [p for p in config.plugins.plugins
                                  if p.name != self.to_unplug]


class NipyNoseTester(NoseTester):
    """ Numpy-like testing class

    Unlike the numpy version, does not replace the normal doctest plugin with
    the numpy doctest plugin.  Does not exclude some numpy-specific paths from
    testing.
    """
    test_program_maker = NipyTestProgram

    def prepare_test_args(self, label='fast', verbose=1, extra_argv=None,
                          doctests=False, coverage=False):
        """ Prepare arguments for testing run

        This is a stripped down version of the numpy method, in order not to
        swap standard doctests (which we do want) for numpy doctests (which we
        don't).

        For parameters see numpy.testing.nosetester.NoseTester.prepare_test_args

        Returns
        -------
        argv : list
            arguments to pass to NumpyTestProgram nose wrapper
        plugins : list
            initialized nose plugins for NumpyTestProgram
        """
        # Code pasted from numpy.testing.nosetester.NoseTester class
        argv = self._test_argv(label, verbose, extra_argv)
        if doctests and not '--with-doctest' in argv:
            argv += ['--with-doctest']
        if coverage:
            argv+=['--cover-package=%s' % self.package_name, '--with-coverage',
                   '--cover-tests', '--cover-inclusive', '--cover-erase']
        nose = import_nose()
        # construct list of plugins
        import nose.plugins.builtin
        from numpy.testing.noseclasses import KnownFailure
        plugins = [KnownFailure()]
        plugins += [p() for p in nose.plugins.builtin.plugins]
        return argv, plugins

    def test(self, label='fast', verbose=1, extra_argv=None, doctests=True,
             coverage=False):
        """
        Run tests for module using nose.

        Identical to numpy version, but using our own TestProgram class, and
        with `doctests` defaulting to True

        Parameters
        ----------
        label : {'fast', 'full', '', attribute identifier}, optional
            Identifies the tests to run. This can be a string to pass to the
            nosetests executable with the '-A' option, or one of
            several special values.  Special values are:
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
            If True, run doctests in module. Default is True
        coverage : bool, optional
            If True, report coverage of NumPy code. Default is False.
            (This requires the `coverage module:
             <http://nedbatchelder.com/code/modules/coverage.html>`_).

        Returns
        -------
        result : object
            Returns the result of running the tests as a
            ``nose.result.TextTestResult`` object.

        Notes
        -----
        Each NumPy module exposes `test` in its namespace to run all tests for it.
        For example, to run all tests for numpy.lib::

          >>> np.lib.test()

        Examples
        --------
        >>> result = np.lib.test()
        Running unit tests for numpy.lib
        ...
        Ran 976 tests in 3.933s

        OK

        >>> result.errors
        []
        >>> result.knownfail
        []
        """
        # cap verbosity at 3 because nose becomes *very* verbose beyond that
        verbose = min(verbose, 3)

        from numpy.testing import utils
        utils.verbose = verbose

        if doctests:
            print "Running unit tests and doctests for %s" % self.package_name
        else:
            print "Running unit tests for %s" % self.package_name

        self._show_system_info()

        # reset doctest state on every run
        import doctest
        doctest.master = None

        argv, plugins = self.prepare_test_args(label, verbose, extra_argv,
                                               doctests, coverage)
        t = self.test_program_maker(argv=argv, exit=False, plugins=plugins)
        return t.result

    def bench(self, label='fast', verbose=1, extra_argv=None):
        """
        Run benchmarks for module using nose.

        This version is from numpy, and adds a patch to stop this routine
        running doctests even if enabled in the user's nose configuration.

        Parameters
        ----------
        label : {'fast', 'full', '', attribute identifier}, optional
            Identifies the benchmarks to run. This can be a string to pass to the
            nosetests executable with the '-A' option, or one of
            several special values.
            Special values are:
            * 'fast' - the default - which corresponds to the ``nosetests -A``
              option of 'not slow'.
            * 'full' - fast (as above) and slow benchmarks as in the
              'no -A' option to nosetests - this is the same as ''.
            * None or '' - run all tests.
            attribute_identifier - string passed directly to nosetests as '-A'.
        verbose : int, optional
            Verbosity value for test outputs, in the range 1-10. Default is 1.
        extra_argv : list, optional
            List with any extra arguments to pass to nosetests.

        Returns
        -------
        success : bool
            Returns True if running the benchmarks works, False if an error
            occurred.

        Notes
        -----
        Benchmarks are like tests, but have names starting with "bench" instead
        of "test", and can be found under the "benchmarks" sub-directory of the
        module.

        Each NumPy module exposes `bench` in its namespace to run all benchmarks
        for it.

        Examples
        --------
        >>> success = np.lib.bench()
        Running benchmarks for numpy.lib
        ...
        using 562341 items:
        unique:
        0.11
        unique1d:
        0.11
        ratio: 1.0
        nUnique: 56230 == 56230
        ...
        OK

        >>> success
        True

        """
        # get plugin to disable doctests
        add_plugins = [Unplugger('doctest')]

        print "Running benchmarks for %s" % self.package_name
        self._show_system_info()

        argv = self._test_argv(label, verbose, extra_argv)
        argv += ['--match', r'(?:^|[\\b_\\.%s-])[Bb]ench' % os.sep]

        nose = import_nose()
        return nose.run(argv=argv, addplugins=add_plugins)
