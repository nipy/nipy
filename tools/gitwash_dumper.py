#!/usr/bin/env python
''' Checkout gitwash repo into directory and do search replace on name '''

import os
import shutil
import sys
import re
import glob
import fnmatch
import tempfile
import functools
from subprocess import call

caller = functools.partial(call, shell=True)


def clone_repo(url):
    tmpdir = tempfile.mkdtemp()
    caller('git clone %s %s' % (url, tmpdir))
    return tmpdir


def cp_files(in_path, globs, out_path):
    try:
        os.makedirs(out_path)
    except OSError:
        pass
    out_fnames = []
    for in_glob in globs:
        in_glob_path = os.path.join(in_path, in_glob)
        for in_fname in glob.glob(in_glob_path):
            out_fname = in_fname.replace(in_path, out_path)
            pth, _ = os.path.split(out_fname)
            if not os.path.isdir(pth):
                os.makedirs(pth)
            print out_fname
            shutil.copyfile(in_fname, out_fname)
            out_fnames.append(out_fname)
    return out_fnames


def perl_dash_pie(in_exp, out_str, filename):
    in_reg = re.compile(in_exp)
    in_txt = open(filename, 'rt').read(-1)
    out_txt = in_reg.sub(out_str, in_txt)
    if in_txt != out_txt:
        open(filename, 'wt').write(out_txt)
        return True
    return False

    
def copy_replace(project_name, out_path, repo_url, replace_str,
                 cp_globs=('gitwash/*.txt', 'gitwash/*.rst'),
                 rep_globs=('*.rst',)):
    repo_path = clone_repo(gitwash_url)
    out_fnames = cp_files(repo_path, cp_globs, out_path)
    shutil.rmtree(repo_path)
    fnames = []
    for rep_glob in rep_globs:
        fnames += fnmatch.filter(out_fnames, rep_glob)
    for fname in fnames:
        perl_dash_pie(replace_str, project_name, fname)


usage = ('Usage: '
         '%s <project_name> <out_path> [gitwash-url]')


if __name__ == '__main__':
    prog = sys.argv.pop(0)
    if len(sys.argv) < 2:
        raise OSError(usage % prog)
    project_name = sys.argv.pop(0)
    out_path = sys.argv.pop(0)
    if len(sys.argv):
        gitwash_url = sys.argv.pop(0)
    else:
        gitwash_url = 'git://github.com/matthew-brett/gitwash.git'
    copy_replace(project_name, out_path, gitwash_url, 'ipython')
    
