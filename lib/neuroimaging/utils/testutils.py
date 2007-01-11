"""
Module containing simple utilities for running our tests.
"""
import shutil, tempfile, os, types
from unittest import TestSuite, makeSuite, TestCase, TextTestRunner
from doctest import DocFileSuite
from glob import glob
from os.path import join, dirname, basename

from neuroimaging import nontest_packages, import_from
from neuroimaging.utils.odict import odict

def run_suite(suite):
    tmpdir = tempfile.mkdtemp(suffix='nipy_unittest')
    curdir = os.path.abspath(os.curdir)
    os.chdir(tmpdir)
    TextTestRunner(verbosity=2).run(suite)
    os.chdir(curdir)
    shutil.rmtree(tmpdir, ignore_errors=False)

def run_tests(*tests):
    for test in tests:
        print test
    run_suite(TestSuite([makeSuite(test) for test in tests]))

def get_package_modules(package):
    modfiles = glob(join(dirname(package.__file__), "*.py"))
    modules = []
    for modfile in modfiles:
        modname = basename(modfile).split(".")[0]
        full_modname = "%s.%s"%(package.__name__,modname)
        modules.append(__import__(full_modname,{},{},[modname]))
    return modules

def get_package_doctests(packname):
    modname = packname.split(".")[-1]
    try:
        package = __import__(packname,{},{},[modname])
        suites = ()
        for module in get_package_modules(package):
            pyfile = '%s.py' % os.path.splitext(module.__file__)[0]
            suites += (DocFileSuite(pyfile,
                                    module_relative=False),)
        return TestSuite(suites)
    except:
        return TestSuite()

def get_package_tests(packname):
    """
    Retrieve all TestCases defined for the given package.
    @returns {testname:testclass,...}
    """

    package = import_from(packname, "tests")
    if not package: return {}
    modules = get_package_modules(package)
    tests = []
    for module in modules:
        tests += [(key,val) for key,val in module.__dict__.items() \
                  if isinstance(val, type) and \
                     issubclass(val, TestCase) and \
                     val != TestCase]
    return dict(tests)
    
def test_package(packname, testname=None):
    "Run all tests for the given package"
    testdict = get_package_tests(packname)
    tests = [val for key,val in testdict.items() \
             if testname==None or testname==key]
    run_tests(*tests)

def test_packages(*packages):
    "Run all tests for the given packages"
    tests = odict()
    for packname in packages: tests.update(get_package_tests(packname))
    tests.sort()
    run_tests(*tests.values())

def test_all(): test_packages(*nontest_packages)

def doctest_package(packname):
    "Run all doctests for the given package"
    suite = get_package_doctests(packname)
    if suite:
        run_suite(suite)

def doctest_packages(*packages):
    "Run all doctests for the given packages"
    tests = []
    for packname in packages:
        suite = get_package_doctests(packname)
        if suite: tests += suite

    run_suite(TestSuite(tests))

def doctest_all(): doctest_packages(*nontest_packages)
