import unittest, shutil, tempfile, os

packages = (
    'neuroimaging',
    'neuroimaging.statistics',
    'neuroimaging.image',
    'neuroimaging.reference',
    'neuroimaging.data',
    'neuroimaging.image.formats',
    'neuroimaging.image.formats.analyze',
    'neuroimaging.fmri',
    'neuroimaging.fmri.fmristat',
    'neuroimaging.visualization',
    'neuroimaging.visualization.cmap')


def import_from(modulename, objectname):
    "Import and return objectname from modulename."
    module = __import__(modulename, globals(), locals(), (objectname,))
    return getattr(module, objectname)

def get_package_suite(packname):
    """
    Retrieve the test suite for the named package.  Expects to find under the
    named package another package or module named 'tests', containing a
    a callable called 'suite' returning a unittest.TestSuite object.
    """
    packname = packname + ".tests"
    try:
        return import_from(packname, "suite")()
    except ImportError:
        raise ValueError("Could not find expected test package '%s'."%packname)

def run_suite(suite):
    tmpdir = tempfile.mkdtemp(suffix='nipy_unittest')
    curdir = os.path.abspath(os.curdir)
    os.chdir(tmpdir)
    unittest.TextTestRunner(verbosity=2).run(suite)
    os.chdir(curdir)
    shutil.rmtree(tmpdir, ignore_errors=False)
    
def test_package(packname):
    try:
        suite = get_package_suite(packname)
    except Exception, e:
        print "Failed to run test suite for package '%s' because: %s"\
          %(packname,e)
        return 
    run_suite(get_package_suite(packname))

def test_all():
    for packname in packages:
        test_package(packname)

def suite(): return unittest.TestSuite()
