#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Script to make a copy of pynifti / brifti for insertion into NIPY tree

It downloads a copy from the repository, patches the paths, tests, makes
an archive, deletes temporary copies, and exits.

Unix specific.  It could be generalized, but it's only for occasional
use in updating the code tree, and probably won't be used for long
either.

Typical use would be (to port over changes for revision tagged in git as
'brifti-0.4':

copy_brifti.py /tmp/brifti-0.4.tar.gz brifti-0.4
cd ~/nipy-repo/trunk-lp/nipy/io
tar zcvf /tmp/brifti-0.4.tar.gz

then bzr stat, commit as necessary.

Or, when working locally, something like:

copy_brifti.py /tmp/brifti-0.4.tar.gz brifti-0.4 ~/dev_trees/pynifti/.git

to replace the first line above.
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

caller = functools.partial(call, shell=True)
git_path = 'git://github.com/hanke/nibabel.git'


# search replaces for imports
def import_replace(pth, old_import, new_import):
    subs = (
        (re.compile(r'^([ >]*)(import|from) +%s' % old_import),
         r'\1\2 %s' % new_import),
        )
    for root, dirs, files in os.walk(pth):
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


def create_archive(out_path, git_path, git_id,
                   dist_sdir, pkg_name):
    out_path = os.path.abspath(out_path)
    pwd = os.path.abspath(os.curdir)
    # pull out git archive
    tmp_path = tempfile.mkdtemp()
    os.chdir(tmp_path)
    tar_name = pjoin(tmp_path, '%s.tar' % pkg_name)
    caller('git clone ' + git_path)
    # extract nifti library tree from git archive
    os.chdir(dist_sdir)
    caller('git archive %s %s > %s' %
           (git_id, pkg_name, tar_name))
    os.chdir(tmp_path)
    shutil.rmtree(dist_sdir)
    caller('tar xvf %s' % tar_name)
    # create path structure for moved imports
    os.makedirs(pjoin('nipy', 'io'))
    file(pjoin('nipy', '__init__.py'), 'wt').write('\n')
    file(pjoin('nipy', 'io', '__init__.py'), 'wt').write('\n')
    os.rename(pkg_name, pjoin('nipy','io','imageformats'))
    # do search and replace for imports to change to NIPY ones
    import_replace('nipy', 'nibabel', 'nipy.io.imageformats')
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
        git_id = 'nipy-io-0.5'
    try:
        git_path = sys.argv[3]
    except IndexError:
        pass
    create_archive(out_path, git_path, git_id, 'nibabel', 'nibabel')
