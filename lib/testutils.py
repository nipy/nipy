import shutil, tempfile, os, types
from unittest import TestSuite, makeSuite, TestCase, TextTestRunner
from glob import glob
from os.path import join, dirname, basename

from odict import odict

from neuroimaging import nontest_packages, import_from

#-----------------------------------------------------------------------------
def run_suite(suite):
    tmpdir = tempfile.mkdtemp(suffix='nipy_unittest')
    curdir = os.path.abspath(os.curdir)
    os.chdir(tmpdir)
    TextTestRunner(verbosity=2).run(suite)
    os.chdir(curdir)
    shutil.rmtree(tmpdir, ignore_errors=False)

#-----------------------------------------------------------------------------
def run_tests(*tests):
    for test in tests:
        print test
    run_suite(TestSuite([makeSuite(test) for test in tests]))

#-----------------------------------------------------------------------------
def get_package_modules(package):
    modfiles = glob(join(dirname(package.__file__), "*.py"))
    modules = []
    for modfile in modfiles:
        modname = basename(modfile).split(".")[0]
        full_modname = "%s.%s"%(package.__name__,modname)
        modules.append(__import__(full_modname,{},{},[modname]))
    return modules

#-----------------------------------------------------------------------------
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
                  if type(val)==types.TypeType and \
                     issubclass(val, TestCase) and \
                     val != TestCase]
    return dict(tests)
    
#-----------------------------------------------------------------------------
def test_package(packname, testname=None):
    "Run all tests for the given package"
    testdict = get_package_tests(packname)
    tests = [val for key,val in testdict.items() \
             if testname==None or testname==key]
    run_tests(*tests)

#-----------------------------------------------------------------------------
def test_packages(*packages):
    "Run all tests for the given packages"
    tests = odict()
    for packname in packages: tests.update(get_package_tests(packname))
    tests.sort()
    run_tests(*tests.values())

#-----------------------------------------------------------------------------
def test_all(): test_packages(*nontest_packages)
