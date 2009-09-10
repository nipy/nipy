
import os
from tempfile import mkdtemp
from shutil import rmtree

from nipy.testing import *
from sneeze import find_pkg, run_nose

import_strings = ["from nipype.interfaces.afni import To3d, ThreeDRefit",
                  "from nipype.interfaces import afni",
                  "import nipype.interfaces.afni",
                  "from nipype.interfaces import afni as af"]

def test_imports():
    dname = mkdtemp()
    fname = os.path.join(dname, 'test_afni.py')
    for impt in import_strings:
        fp = open(fname, 'w')
        fp.write(impt)
        fp.close()
        cover_pkg, module = find_pkg(fname)
        cmd = run_nose(cover_pkg, fname, dry_run=True)
        cmdlst = cmd.split()
        cmd = ' '.join(cmdlst[:4]) # strip off temporary directory path
        yield assert_equal, cmd, \
        'nosetests -sv --with-coverage --cover-package=nipype.interfaces.afni'
    if os.path.exists(dname):
        rmtree(dname)
