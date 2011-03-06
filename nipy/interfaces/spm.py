# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Interfaces to SPM '''
from __future__ import with_statement

import os

import numpy as np

from scipy.io import savemat

from nipy.utils import InTemporaryDirectory, setattr_on_read
from nibabel import load
from nipy.interfaces.matlab import run_matlab_script


class SpmInfo(object):
    @setattr_on_read
    def spm_path(self):
        with InTemporaryDirectory() as tmpdir:
            run_matlab_script("""
spm_path = spm('dir');
fid = fopen('spm_path.txt', 'wt');
fprintf(fid, '%s', spm_path);
fclose(fid);
""")
            spm_path = file('spm_path.txt', 'rt').read()
        return spm_path

spm_info = SpmInfo()

                
def make_job(jobtype, jobname, contents):
    return {'jobs':[{jobtype:[{jobname:contents}]}]}


# XXX this should be moved into a matdict class or something
def fltcols(vals):
    ''' Trivial little function to make 1xN float vector '''
    return np.atleast_2d(np.array(vals, dtype=float))


def run_jobdef(jobdef):
    with InTemporaryDirectory():
        savemat('pyjobs.mat', jobdef)
        run_matlab_script("""
load pyjobs;
spm_jobman('run', jobs);
""")


def scans_for_fname(fname):
    img = load(fname)
    n_scans = img.get_shape()[3]
    scans = np.zeros((n_scans, 1), dtype=object)
    for sno in range(n_scans):
        scans[sno] = '%s,%d' % (fname, sno+1)
    return scans


def scans_for_fnames(fnames):
    n_sess = len(fnames)
    sess_scans = np.zeros((1,n_sess), dtype=object)
    for sess in range(n_sess):
        sess_scans[0,sess] = scans_for_fname(fnames[sess])
    return sess_scans


def fname_presuffix(fname, prefix='', suffix='', use_ext=True):
    pth, fname = os.path.split(fname)
    fname, ext = os.path.splitext(fname)
    if not use_ext:
        ext = ''
    return os.path.join(pth, prefix+fname+suffix+ext)


def fnames_presuffix(fnames, prefix='', suffix=''):
    f2 = []
    for fname in fnames:
        f2.append(fname_presuffix(fname, prefix, suffix))
    return f2


