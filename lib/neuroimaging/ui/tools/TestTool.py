"""
Simple utility for running unit tests.
"""

import sys
from optparse import OptionParser, Option

from neuroimaging.utils.testutils import test_all, test_package,  \
    get_package_test, doctest_all, doctest_package
from neuroimaging import nontest_packages

# description = __doc__


class TestTool (OptionParser):
    """
    %prog [options] [<pkgname> [<testname>]]

    Tests will be run for the named package (or all packages).  If a test name
    is given, only that test will be run.  If the list option is given,
    avalailable tests will be listed for the given package (or all packages).
    """
    
    _usage = __doc__
    options = (
      Option('-l', '--list', dest="list_tests", action="store_true",
        default=False, help="list available tests"),
      Option('-d', '--doctest', dest="doctest", action="store_true",
        default=False, help="run available doctests"),
      Option('', '--nounit', dest="unit", action="store_false",
        default=True, help="do not run unittests"),)


    def __init__(self, *args, **kwargs):
        OptionParser.__init__(self, *args, **kwargs)
        self.set_usage(self._usage)
        self.add_options(self.options)
#        self.set_description("description:" + description)


    def list_tests(self, package=None):
        if package:
            packs = [package]
        else:
            packs = list(nontest_packages)
            packs.sort()
        for package in packs:
            tests = get_package_tests(package).keys()
            tests.sort()
            print package,":"
            if tests:
                for test in tests: print " ",test
            else:
                print "No tests found for this package"

    def run_tests(self, package=None, testcase=None):
        if not package: test_all()
        else: test_package(package, testcase)

    def run_doctests(self, package=None):
        if not package: doctest_all()
        else: doctest_package(package)

    def run(self):
        options, args = self.parse_args()

        if options.list_tests:
            if len(args) > 1:
                self.print_help()
                sys.exit(0)
            self.list_tests(*args)
        elif options.unit: self.run_tests(*args)

        if options.doctest and len(args)<=1:
            self.run_doctests(*args)

if __name__ == "__main__": TestTool().run()
