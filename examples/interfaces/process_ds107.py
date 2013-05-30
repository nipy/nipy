#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Single subject analysis script for SPM / Open FMRI ds107 '''
import sys
from copy import deepcopy
from os.path import join as pjoin, abspath, splitext, isfile
from glob import glob
from warnings import warn
import gzip

import numpy as np

import nipy.interfaces.matlab as nimat
from nipy.interfaces.spm import (spm_info, make_job, scans_for_fnames,
                                 run_jobdef, fnames_presuffix, fname_presuffix,
                                 fltcols)


# The batch scripts currently need SPM5
nimat.matlab_cmd = 'matlab-2007a-spm5 -nodesktop -nosplash'

N_SLICES = 37
STUDY_DEF = dict(
    TR = 3.0,
    n_slices = N_SLICES,
    time_to_space = range(1, N_SLICES, 2) + range(2, N_SLICES, 2)
)


def _sorted_prefer_nii(file_list):
    """ Strip any filanames ending nii.gz if matching .nii filename in list
    """
    preferred = []
    for fname in file_list:
        if not fname.endswith('.gz'):
            preferred.append(fname)
        else:
            nogz, ext = splitext(fname)
            if not nogz in file_list:
                preferred.append(fname)
    return sorted(preferred)


def get_data(data_path, subj_id):
    data_path = abspath(data_path)
    data_def = {}
    subject_path = pjoin(data_path, 'sub%03d' % subj_id)
    functionals = _sorted_prefer_nii(
        glob(pjoin(subject_path, 'BOLD', 'task*', 'bold*.nii*')))
    anatomicals = _sorted_prefer_nii(
        glob(pjoin(subject_path, 'anatomy', 'highres001.nii*')))
    for flist in (anatomicals, functionals):
        for i, fname in enumerate(flist):
            nogz, gz_ext = splitext(fname)
            if gz_ext == '.gz':
                if not isfile(nogz):
                    contents = gzip.open(fname, 'rb').read()
                    with open(nogz, 'wb') as fobj:
                        fobj.write(contents)
                flist[i] = nogz
    data_def['anatomical'] = anatomicals[0]
    data_def['functionals'] = functionals
    return data_def


def default_ta(tr, nslices):
    slice_time = tr / float(nslices)
    return slice_time * (nslices - 1)


class SPMSubjectAnalysis(object):
    """ Class to preprocess single subject in SPM
    """
    def __init__(self, data_def, study_def, ana_def):
        self.data_def = deepcopy(data_def)
        self.study_def = self.add_study_defaults(study_def)
        self.ana_def = self.add_ana_defaults(deepcopy(ana_def))

    def add_study_defaults(self, study_def):
        full_study_def = deepcopy(study_def)
        if 'TA' not in full_study_def:
            full_study_def['TA'] = default_ta(
                full_study_def['TR'], full_study_def['n_slices'])
        return full_study_def

    def add_ana_defaults(self, ana_def):
        full_ana_def = deepcopy(ana_def)
        if 'fwhm' not in full_ana_def:
            full_ana_def['fwhm'] = 8.0
        return full_ana_def

    def slicetime(self, prefix=''):
        sess_scans = scans_for_fnames(self.data_def['functionals'])
        sdef = self.study_def
        stinfo = make_job('temporal', 'st', {
                'scans': sess_scans,
                'so': sdef['time_to_space'],
                'tr': sdef['TR'],
                'ta': sdef['TA'],
                'nslices': float(sdef['n_slices']),
                'refslice':1
                })
        run_jobdef(stinfo)
        return 'a' + prefix


    def realign(self, prefix=''):
        sess_scans = scans_for_fnames(
            fnames_presuffix(self.data_def['functionals'], 'a'))
        rinfo = make_job('spatial', 'realign', [{
                'estimate':{
                    'data':sess_scans,
                    'eoptions':{
                        'quality': 0.9,
                        'sep': 4.0,
                        'fwhm': 5.0,
                        'rtm': True,
                        'interp': 2.0,
                        'wrap': [0.0,0.0,0.0],
                        'weight': []
                        }
                    }
                }])
        run_jobdef(rinfo)
        return prefix

    def reslice(self, prefix=''):
        sess_scans = scans_for_fnames(
            fnames_presuffix(self.data_def['functionals'], prefix))
        rsinfo = make_job('spatial', 'realign', [{
                'write':{
                    'data': np.vstack(sess_scans.flat),
                    'roptions':{
                        'which':[2, 1],
                        'interp':4.0,
                        'wrap':[0.0,0.0,0.0],
                        'mask':True,
                        }
                    }
                }])
        run_jobdef(rsinfo)
        return 'r' + prefix

    def coregister(self, prefix=''):
        func1 = self.data_def['functionals'][0]
        mean_fname = fname_presuffix(func1, 'mean' + prefix)
        crinfo = make_job('spatial', 'coreg', [{
                'estimate':{
                    'ref': [mean_fname],
                    'source': [self.data_def['anatomical']],
                    'other': [[]],
                    'eoptions':{
                        'cost_fun':'nmi',
                        'sep':[4.0, 2.0],
                        'tol':np.array(
                                [0.02,0.02,0.02,
                                0.001,0.001,0.001,
                                0.01,0.01,0.01,
                                0.001,0.001,0.001]).reshape(1,12),
                        'fwhm':[7.0, 7.0]
                        }
                    }
                }])
        run_jobdef(crinfo)
        return prefix

    def segnorm(self, prefix=''):
        def_tpms = np.zeros((3,1), dtype=np.object)
        spm_path = spm_info.spm_path
        def_tpms[0] = pjoin(spm_path, 'tpm', 'grey.nii'),
        def_tpms[1] = pjoin(spm_path, 'tpm', 'white.nii'),
        def_tpms[2] = pjoin(spm_path, 'tpm', 'csf.nii')
        data = np.zeros((1,), dtype=object)
        data[0] = self.data_def['anatomical']
        sninfo = make_job('spatial', 'preproc', {
                'data': data,
                'output':{
                    'GM':fltcols([0,0,1]),
                    'WM':fltcols([0,0,1]),
                    'CSF':fltcols([0,0,0]),
                    'biascor':1.0,
                    'cleanup':False,
                    },
                'opts':{
                    'tpm':def_tpms,
                    'ngaus':fltcols([2,2,2,4]),
                    'regtype':'mni',
                    'warpreg':1.0,
                    'warpco':25.0,
                    'biasreg':0.0001,
                    'biasfwhm':60.0,
                    'samp':3.0,
                    'msk':np.array([], dtype=object),
                    }
                })
        run_jobdef(sninfo)
        return prefix

    def norm_write(self, prefix=''):
        sess_scans = scans_for_fnames(
            fnames_presuffix(self.data_def['functionals'], prefix))
        matname = fname_presuffix(self.data_def['anatomical'],
                                suffix='_seg_sn.mat',
                                use_ext=False)
        subj = {
            'matname': np.zeros((1,), dtype=object),
            'resample': np.vstack(sess_scans.flat),
            }
        subj['matname'][0] = matname
        roptions = {
            'preserve':False,
            'bb':np.array([[-78,-112, -50],[78,76,85.0]]),
            'vox':fltcols([2.0,2.0,2.0]),
            'interp':1.0,
            'wrap':[0.0,0.0,0.0],
            }
        nwinfo = make_job('spatial', 'normalise', [{
                'write':{
                    'subj': subj,
                    'roptions': roptions,
                    }
                }])
        run_jobdef(nwinfo)
        # knock out the list of images, replacing with only one
        subj['resample'] = np.zeros((1,), dtype=object)
        subj['resample'][0] = self.data_def['anatomical']
        roptions['interp'] = 4.0
        run_jobdef(nwinfo)
        return 'w' + prefix

    def smooth(self, prefix=''):
        fwhm = self.ana_def['fwhm']
        try:
            len(fwhm)
        except TypeError:
            fwhm = [fwhm] * 3
        fwhm = np.asarray(fwhm, dtype=np.float).reshape(1,3)
        sess_scans = scans_for_fnames(
            fnames_presuffix(self.data_def['functionals'], prefix))
        sinfo = make_job('spatial', 'smooth',
                        {'data':np.vstack(sess_scans.flat),
                        'fwhm':fwhm,
                        'dtype':0})
        run_jobdef(sinfo)
        return 's' + prefix


def process_subject(ddef, study_def, ana_def):
    """ Process subject from subject data dict `ddef`
    """
    if not ddef['anatomical']:
        warn("No anatomical, aborting processing")
        return
    ana = SPMSubjectAnalysis(ddef, study_def, ana_def)
    st_prefix = ana.slicetime('')
    ana.realign(st_prefix)
    ana.coregister(st_prefix)
    ana.segnorm()
    n_st_prefix = ana.norm_write(st_prefix)
    ana.smooth(n_st_prefix)


def get_subjects(data_path, subj_ids, study_def, ana_def):
    ddefs = []
    for subj_id in subj_ids:
        ddefs.append(get_data(data_path, subj_id))
    return ddefs


if __name__ == '__main__':
    try:
        data_path = sys.argv[1]
    except IndexError:
        raise OSError('Need ds107 data path as input')
    if len(sys.argv) > 2:
        subj_ids = [int(id) for id in sys.argv[2:]]
    else:
        subj_ids = range(1, 16)
    for subj_id in subj_ids:
        ddef = get_data(data_path, subj_id)
        assert len(ddef['functionals']) == 2
        assert len(ddef['anatomicals']) == 1
        process_subject(ddef, STUDY_DEF, {})
