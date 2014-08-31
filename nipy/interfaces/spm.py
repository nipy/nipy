# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Interfaces to SPM '''
from __future__ import with_statement

import os

import numpy as np

from scipy.io import savemat

from nibabel import load
from nibabel.tmpdirs import InTemporaryDirectory

from .matlab import run_matlab_script


class SpmInfo(object):
    def __init__(self):
        self._spm_path = None
        self._spm_ver = None

    def _set_properties(self):
        with InTemporaryDirectory():
            run_matlab_script(r"""
spm_path = spm('dir');
spm_ver = spm('ver');
fid = fopen('spm_stuff.txt', 'wt');
fprintf(fid, '%s\n', spm_path);
fprintf(fid, '%s\n', spm_ver);
fclose(fid);
""")
            with open('spm_stuff.txt', 'rt') as fobj:
                lines = fobj.readlines()
        self._spm_path = lines[0].strip()
        self._spm_ver = lines[1].strip()

    @property
    def spm_path(self):
        if self._spm_path is None:
            self._set_properties()
        return self._spm_path

    @property
    def spm_ver(self):
        if self._spm_ver is None:
            self._set_properties()
        return self._spm_ver


spm_info = SpmInfo()


def make_job(jobtype, jobname, contents):
    return {'jobs':[{jobtype:[{jobname:contents}]}]}


# XXX this should be moved into a matdict class or something
def fltcols(vals):
    ''' Trivial little function to make 1xN float vector '''
    return np.atleast_2d(np.array(vals, dtype=float))


def run_jobdef(jobdef):
    script = """
load pyjobs;
spm_jobman('run', jobs);
"""
    # Need initcfg for SPM8
    if spm_info.spm_ver != 'SPM5':
        script = "spm_jobman('initcfg');\n" + script
    with InTemporaryDirectory():
        savemat('pyjobs.mat', jobdef, oned_as='row')
        run_matlab_script(script)


def scans_for_fname(fname):
    img = load(fname)
    n_scans = img.shape[3]
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


