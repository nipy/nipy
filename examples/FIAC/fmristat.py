import os, urllib2
from csv import reader

from scipy.io import loadmat
from numpy import asarray

import io
from neuroimaging.core.api import Image

contrast_map = {'sentence': 'sen',
                'speaker': 'spk',
                'average': 'all',
                'interaction': 'snp'}

which_map = {'contrasts': 'mag',
             'delays': 'del'}

stat_map = {'t':'t',
            'effect': 'ef',
            'sd': 'sd'}

design_map = {'event':'evt',
              'block':'bloc'}

def rho(subject=3, run=3):
    """
    Estimate of AR(1) coefficient from fmristat
    """
    runfile = '%s/fmristat/fiac%d/fiac%d_fonc%d_all_cor.img' % (io.web_path, subject, subject, run)
    return Image(runfile)

def result(subject=3, run=3, which='contrasts', contrast='average', stat='t'):
    """
    Retrieve an fmristat result for one run of the FIAC data.
    """
    contrast = contrast_map[contrast]
    which = which_map[which]
    stat = stat_map[stat]

    resultfile = '%s/fmristat/fiac%d/fiac%d_fonc%d_%s_%s_%s.img' % (io.web_path, subject, subject, run, contrast, which, stat)
    return Image(resultfile)

def fixed(subject=3, which='contrasts', contrast='average', design='block', stat='effect'):
    """
    Retrieve a within-subject fixed effect fmristat result for one FIAC subject.
    """
    contrast = contrast_map[contrast]
    which = which_map[which]
    stat = stat_map[stat]
    design = design_map[design]

    resultfile = '%s/fmristat/subj/subj%d_%s_%s_%s_%s.img' % (io.web_path, subject, design, contrast, which, stat)

    return Image(resultfile)

def multi(which='contrasts', contrast='average', design='block', stat='effect'):
    """
    Retrieve a random effect fmristat result.
    """
    contrast = contrast_map[contrast]
    which = which_map[which]
    stat = stat_map[stat]
    design = design_map[design]

    resultfile = '%s/fmristat/multi/multi_%s_%s_%s_%s.img' % (io.web_path, design, contrast, which, stat)
    return Image(resultfile)

def xcache(subj=0, run=1):
    """
    fmristat X_cache
    """
    
    x_cache = 'x_cache/subj%d_run%d.mat' % (subj, run)
    if not os.path.exists(x_cache):
        url = '%s/x_cache/mat/subj%d_run%d.mat' % (io.web_path, subj, run)
        mat = urllib2.urlopen(url).read()
        if not os.path.exists('x_cache'):
            os.makedirs('x_cache')
        outfile = file('x_cache/subj%d_run%d.mat' % (subj, run), 'wb')
        outfile.write(mat)
        outfile.close()

    X = loadmat(x_cache)['X_cache'].X
    if len(X.shape) == 4:
        X = X[:,:,:,0]
    return X

def design(subj=0, run=1):
    """
    fmristat design matrix
    """

    design = "x_cache/subj%d_run%d.csv" % (subj, run)
    if not os.path.exists(design):
        url = '%s/x_cache/mat/subj%d_run%d.csv' % (io.web_path, subj, run)
        mat = urllib2.urlopen(url).read()
        if not os.path.exists('x_cache'):
            os.makedirs('x_cache')
        outfile = file('x_cache/subj%d_run%d.csv' % (subj, run), 'wb')
        outfile.write(mat)
        outfile.close()

    out = []
    for row in reader(file(design), delimiter='\t'):
        out.append([float(r) for r in row[:15]])
    return asarray(out)
