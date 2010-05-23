#!/usr/bin/env python
''' Checkout gitwash repo into directory and do search replace on name '''

import os
import shutil
import sys
import re
import glob
import fnmatch
import tempfile
from subprocess import call


def clone_repo(url, branch):
    tmpdir = tempfile.mkdtemp()
    cmd = 'git clone --branch %s %s %s' % (branch, url, tmpdir)
    print cmd
    call(cmd, shell=True)
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

    
def copy_replace(replace_pairs,
                 out_path,
                 repo_url,
                 repo_branch = 'master',
                 cp_globs=('gitwash/*',),
                 rep_globs=('*.rst',)):
    repo_path = clone_repo(gitwash_url, repo_branch)
    out_fnames = cp_files(repo_path, cp_globs, out_path)
    shutil.rmtree(repo_path)
    fnames = []
    for rep_glob in rep_globs:
        fnames += fnmatch.filter(out_fnames, rep_glob)
    for fname in fnames:
        for old, new in replace_pairs:
            perl_dash_pie(old, new, fname)


usage = ('Usage: '
         '%s <out_path> <project_name> [<repo_name> '
         '[main_github_user '
         '[gitwash-url [gitwash-branch]]]]')


if __name__ == '__main__':
    prog = sys.argv.pop(0)
    if len(sys.argv) < 2:
        raise OSError(usage % prog)
    out_path, project_name = sys.argv[:2]
    try:
        repo_name = sys.argv[2]
    except IndexError:
        repo_name = project_name
    try:
        main_gh_user = sys.argv[3]
    except IndexError:
        main_gh_user = repo_name
    try:
        gitwash_url = sys.argv[4]
    except IndexError:
        gitwash_url = 'git://github.com/matthew-brett/gitwash.git'
    try:
        gitwash_branch = sys.argv[5]
    except IndexError:
        gitwash_branch = 'master'
    copy_replace((('PROJECTNAME', project_name),
                  ('REPONAME', repo_name),
                  ('MAIN_GH_USER', main_gh_user)),
                 out_path, gitwash_url, gitwash_branch)
    
