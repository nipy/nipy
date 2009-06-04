#!/usr/bin/env python
''' Script to copy a version of pynifti / brifti into NIPY tree '''

import os
import sys
import shutil
import tempfile
import functools
from subprocess import Popen, call, PIPE
import re
import nose

subs = (
    (re.compile(r'^([ >]*)(import|from) +nifti'),
     r'\1\2 nipy.io.imageformats'),
    )

caller = functools.partial(call, shell=True)

outpath = tempfile.mkdtemp()
gitpath = '/home/mb312/dev_trees/pynifti/.git'

os.chdir(outpath)
caller('git clone ' + gitpath)
os.chdir('pynifti')
caller('git archive origin/mb/brifti nifti > nifti.tar')
os.chdir('..')
caller('tar xvf pynifti/nifti.tar')
shutil.rmtree('pynifti')
os.rename('nifti', 'imageformats')

for root, dirs, files in os.walk('imageformats'):
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
        
sys.path.append(os.curdir)
nose.run(argv=['--with-doctest'])
caller('tar zcvf imageformats.tar.gz imageformats')
print outpath
