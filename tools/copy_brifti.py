#!/usr/bin/env python
''' Script to make a copy of pynifti / brifti for insertion into NIPY tree

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
git_path = 'git://git.debian.org/git/pkg-exppsy/pynifti.git'
# because I am working locally for the moment
git_path = '/home/mb312/dev_trees/pynifti/.git'


def create_archive(out_path, git_path, git_id):
    pwd = os.path.abspath(os.curdir)
    # pull out git archive
    tmp_path = tempfile.mkdtemp()
    os.chdir(tmp_path)
    caller('git clone ' + git_path)
    # extract nifti library tree from git archive
    os.chdir('pynifti')
    caller('git archive %s nifti > nifti.tar' % git_id)
    os.chdir(tmp_path)
    caller('tar xvf pynifti/nifti.tar')
    shutil.rmtree('pynifti')
    # create path structure for moved imports
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
    # make archive for later use
    os.chdir(pjoin('nipy','io'))
    caller('tar zcvf %s imageformats' % out_path)
    os.chdir(tmp_path)
    # run tests with new import names, bomb out if any fail
    sys.path.insert(0, os.curdir)
    if not nose.run(argv=[os.curdir, '--with-doctest']):
        os.unlink(out_path)
        raise OSError('Tests failed, please check ' + tmp_path)
    os.chdir(pwd)
    shutil.rmtree(tmp_path)
    

if __name__ == '__main__':
    try:
        out_path = sys.argv[1]
    except IndexError:
        raise OSError('Need archive outpath path as input')
    try:
        git_id = sys.argv[2]
    except IndexError:
        git_id = 'brifti-0.1'
    try:
        git_path = sys.argv[3]
    except IndexError:
        pass
    create_archive(out_path, git_path, git_id)
