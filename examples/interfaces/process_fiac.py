#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Single subject analysis script for SPM / FIAC '''
import sys
from os.path import join as pjoin
from glob import glob
import numpy as np

from nipy.interfaces.spm import (spm_info, make_job, scans_for_fnames,
                                 run_jobdef, fnames_presuffix, fname_presuffix,
                                 fltcols)


def get_data(data_path, subj_id):
    data_def = {}
    subject_path = pjoin(data_path, 'fiac%s' % subj_id)
    data_def['functionals'] = sorted(
        glob(pjoin(subject_path, 'functional_*.nii')))
    anatomicals = glob(pjoin(subject_path, 'anatomical.nii'))
    if len(anatomicals) == 1:
        data_def['anatomical'] = anatomicals[0]
    elif len(anatomicals) == 0:
        data_def['anatomical'] = None
    else:
        raise ValueError('Too many anatomicals')
    return data_def


def slicetime(data_def):
    sess_scans = scans_for_fnames(data_def['functionals'])
    stinfo = make_job('temporal', 'st', {
            'scans': sess_scans,
            'so':range(1,31,2) + range(2,31,2),
            'tr':2.5,
            'ta':2.407,
            'nslices':float(30),
            'refslice':1
            })
    run_jobdef(stinfo)


def realign(data_def):
    sess_scans = scans_for_fnames(fnames_presuffix(data_def['functionals'], 'a'))
    rinfo = make_job('spatial', 'realign', [{
            'estimate':{
                'data':sess_scans,
                'eoptions':{
                    'quality':0.9,
                    'sep':4.0,
                    'fwhm':5.0,
                    'rtm':True,
                    'interp':2.0,
                    'wrap':[0.0,0.0,0.0],
                    'weight':[]
                    }
                }
            }])
    run_jobdef(rinfo)


def reslice(data_def):
    sess_scans = scans_for_fnames(fnames_presuffix(data_def['functionals'], 'a'))
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


def coregister(data_def):
    func1 = data_def['functionals'][0]
    mean_fname = fname_presuffix(func1, 'meana')
    crinfo = make_job('spatial', 'coreg', [{
            'estimate':{
                'ref': [mean_fname],
                'source': [data_def['anatomical']],
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


def segnorm(data_def):
    def_tpms = np.zeros((3,1), dtype=np.object)
    spm_path = spm_info.spm_path
    def_tpms[0] = pjoin(spm_path, 'tpm', 'grey.nii'),
    def_tpms[1] = pjoin(spm_path, 'tpm', 'white.nii'),
    def_tpms[2] = pjoin(spm_path, 'tpm', 'csf.nii')
    data = np.zeros((1,), dtype=object)
    data[0] = data_def['anatomical']
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


def norm_write(data_def):
    sess_scans = scans_for_fnames(fnames_presuffix(data_def['functionals'], 'a'))
    matname = fname_presuffix(data_def['anatomical'],
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
    subj['resample'][0] = data_def['anatomical']
    roptions['interp'] = 4.0
    run_jobdef(nwinfo)


def smooth(data_def, fwhm=8.0):
    try:
        len(fwhm)
    except TypeError:
        fwhm = [fwhm] * 3
    fwhm = np.asarray(fwhm, dtype=np.float).reshape(1,3)
    sess_scans = scans_for_fnames(fnames_presuffix(data_def['functionals'], 'wa'))
    sinfo = make_job('spatial', 'smooth',
                     {'data':np.vstack(sess_scans.flat),
                      'fwhm':fwhm,
                      'dtype':0})
    run_jobdef(sinfo)


def process_subject(ddef):
    if not ddef['anatomical']:
        return
    slicetime(ddef)
    realign(ddef)
    reslice(ddef)
    coregister(ddef)
    segnorm(ddef)
    norm_write(ddef)
    smooth(ddef)


def process_subjects(data_path, subj_ids):
    for subj_id in subj_ids:
        ddef = get_data(data_path, subj_id)
        process_subject(ddef)


if __name__ == '__main__':
    try:
        data_path = sys.argv[1]
    except IndexError:
        raise OSError('Need FIAC data path as input')
    try:
        subj_ids = sys.argv[2:]
    except IndexError:
        subj_ids = range(16)
    process_subjects(data_path, subj_ids)
