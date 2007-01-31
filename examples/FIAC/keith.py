import os, urllib2

from scipy.io import loadmat

from neuroimaging.core.image.image import Image

contrast_map = {'sentence': 'sen',
                'speaker': 'spk',
                'average': 'all',
                'interaction': 'snp'}

which_map = {'contrasts': 'mag',
             'delays': 'del'}

stat_map = {'t':'t',
            'effect': 'ef',
            'sd': 'sd'}

def rho(subject=3, run=3):
    runfile = 'http://kff.stanford.edu/FIAC/fmristat/fiac%d/fiac%d_fonc%d_all_cor.img' % (subject, subject, run)
    return Image(runfile)

def result(subject=3, run=3, which='contrasts', contrast='average', stat='t'):
    contrast = contrast_map[contrast]
    which = which_map[which]
    stat = stat_map[stat]

    resultfile = 'http://kff.stanford.edu/FIAC/fmristat/fiac%d/fiac%d_fonc%d_%s_%s_%s.img' % (subject, subject, run, contrast, which, stat)
    return Image(resultfile)

def _getxcache(subj=0, run=1):
    """
    Retrieve x_cache, downloading .mat file from FIAC results website
    if necessary.
    """
    
    x_cache = 'x_cache/x_cache_sub%d_run%d.mat' % (subj, run)
    if not os.path.exists(x_cache):
        mat = urllib2.urlopen('http://kff.stanford.edu/FIAC/x_cache/mat/x_cache_sub%d_run%d.mat' % (subj, run)).read()
        if not os.path.exists('x_cache'):
            os.makedirs('x_cache')
        outfile = file('x_cache/x_cache_sub%d_run%d.mat' % (subj, run), 'wb')
        outfile.write(mat)
        outfile.close()
    X = loadmat(x_cache)['X']
    if len(X.shape) == 4:
        X = X[:,:,:,0]
    return X

def xcache(subj=0, run=1):
    """
    fmristat design matrix
    """
    
    X = getxcache(subj=subj, run=run)[:,:,2:] * 1.
    X.shape = (191, 10)
    return X
