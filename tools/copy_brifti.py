#!/usr/bin/env python
''' Script to copy a version of pynifti / brifti into NIPY tree

It downloads a copy from the repository, patches the paths, tests, makes
an archive, deletes temporary copies, and exits.

Unix specific.  It could be generalized, but it's only for occasional
use in updating the code tree, and probably won't be used for long either.
'''

import os
from os.path import join as pjoin
import sys
import shutil
import tempfile
import functools
from subprocess import call
import re

import nose

# search replaces for imports
subs = (
    (re.compile(r'^([ >]*)(import|from) +nifti'),
     r'\1\2 nipy.io.imageformats'),
    )

caller = functools.partial(call, shell=True)

gitpath = '/home/mb312/dev_trees/pynifti/.git'

outpath = tempfile.mkdtemp()
os.chdir(outpath)
caller('git clone ' + gitpath)
os.chdir('pynifti')
caller('git archive origin/mb/brifti nifti > nifti.tar')
os.chdir('..')
caller('tar xvf pynifti/nifti.tar')
shutil.rmtree('pynifti')
os.makedirs(pjoin('nipy', 'io'))
file(pjoin('nipy', '__init__.py'), 'wt').write('\n')
file(pjoin('nipy', 'io', '__init__.py'), 'wt').write('\n')
os.rename('nifti', pjoin('nipy','io','imageformats'))

# do search and replace for imports to change to NIPY ones
for root, dirs, files in os.walk('nipy'):
    for fname in files:
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(root, fname)
        lines = file(fpath).readlines()
        outfile = file(fpath, 'wt')
        for line in lines:
            for regexp, repstr in subs:
                if regexp.search(line):
                    line = regexp.sub(repstr, line)
                    continue
            outfile.write(line)
        outfile.close()

# run tests with new import names
sys.path.insert(0, os.curdir)
if not nose.run(argv=[os.curdir, '--with-doctest']):
    raise OSError('Tests failed, please check ' + outpath)

os.chdir('nipy/io')
caller('tar zcvf imageformats.tar.gz imageformats')
print outpath
