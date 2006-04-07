import shutil, tempfile, os, types
from unittest import TestSuite, makeSuite, TestCase, TextTestRunner
from glob import glob
from os.path import join, dirname, basename

from neuroimaging import nontest_packages

#-----------------------------------------------------------------------------
def import_from(modulename, objectname):
    "Import and return objectname from modulename."
    module = __import__(modulename, globals(), locals(), (objectname,))
    try: return getattr(module, objectname)
    except AttributeError: return None

#-----------------------------------------------------------------------------
def run_suite(suite):
    tmpdir = tempfile.mkdtemp(suffix='nipy_unittest')
    curdir = os.path.abspath(os.curdir)
    os.chdir(tmpdir)
    TextTestRunner(verbosity=2).run(suite)
    os.chdir(curdir)
    shutil.rmtree(tmpdir, ignore_errors=False)

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
    run_suite(TestSuite([makeSuite(test) for test in tests]))

#-----------------------------------------------------------------------------
def test_all():
    for packname in nontest_packages: test_package(packname)

