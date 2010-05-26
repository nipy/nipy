#!/usr/bin/env python
''' Checkout gitwash repo into directory and do search replace on name '''

import os
from os.path import join as pjoin
import shutil
import sys
import re
import glob
import fnmatch
import tempfile
from subprocess import call


verbose = False


def clone_repo(url, branch):
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    try:
        cmd = 'git clone %s %s' % (url, tmpdir)
        call(cmd, shell=True)
        os.chdir(tmpdir)
        cmd = 'git checkout %s' % branch
        call(cmd, shell=True)
    except:
        shutil.rmtree(tmpdir)
        raise
    finally:
        os.chdir(cwd)
    return tmpdir


def cp_files(in_path, globs, out_path):
    try:
        os.makedirs(out_path)
    except OSError:
        pass
    out_fnames = []
    for in_glob in globs:
        in_glob_path = pjoin(in_path, in_glob)
        for in_fname in glob.glob(in_glob_path):
            out_fname = in_fname.replace(in_path, out_path)
            pth, _ = os.path.split(out_fname)
            if not os.path.isdir(pth):
                os.makedirs(pth)
            shutil.copyfile(in_fname, out_fname)
            out_fnames.append(out_fname)
    return out_fnames


def filename_search_replace(sr_pairs, filename, backup=False):
    ''' Search and replace for expressions in files

    '''
    in_txt = open(filename, 'rt').read(-1)
    out_txt = in_txt[:]
    for in_exp, out_exp in sr_pairs:
        in_exp = re.compile(in_exp)
        out_txt = in_exp.sub(out_exp, out_txt)
    if in_txt == out_txt:
        return False
    open(filename, 'wt').write(out_txt)
    if backup:
        open(filename + '.bak', 'wt').write(in_txt)
    return True

        
def copy_replace(replace_pairs,
                 out_path,
                 repo_url,
                 repo_branch = 'master',
                 cp_globs=('*',),
                 rep_globs=('*',)):
    repo_path = clone_repo(gitwash_url, repo_branch)
    try:
        out_fnames = cp_files(repo_path, cp_globs, out_path)
    finally:
        shutil.rmtree(repo_path)
    fnames = []
    for rep_glob in rep_globs:
        fnames += fnmatch.filter(out_fnames, rep_glob)
    if verbose:
        print '\n'.join(fnames)
    for fname in fnames:
        filename_search_replace(replace_pairs, fname, False)


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
                 out_path,
                 gitwash_url,
                 gitwash_branch,
                 cp_globs=(pjoin('gitwash', '*'),),
                 rep_globs=('*.rst',))
