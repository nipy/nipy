from processing import *
from BrainSTAT.Utils.miscutils import inv
from BrainSTAT.Utils import gslutils
from BrainSTAT.Modules.Histogram import Histogram
from BrainSTAT.Modules import GLM
from BrainSTAT.Utils.miscutils import LinearInterpolant, StepFunction
from BrainSTAT.Base.Options import Warning
from BrainSTAT.fMRIstat.MultiStat import MultiStat

interpolant = LinearInterpolant

import pylab
from numarray.random_array import *
from numarray.mlab import *

gzipped = True

subjects = []
for subj in range(16):
    subject = FIAC(subj)
    subjects.append(subject)

def across_subjects(subjects, contrast, which='block'):
    eff_files = []
    sd_files = []
    df = []

    multidir = '/usr/local/FIAC/multi'
    if not os.path.exists('/usr/local/FIAC/multi'):
        os.mkdir(multidir)

    os.chdir(multidir)

    subjects.pop(2)

    for subject in subjects:
        try:
            eff_files.append(BrainSTAT.VImage(subject.fixedname('eff', contrast, which=which, gzipped=gzipped, resampled=True)))
            sd_files.append(BrainSTAT.VImage(subject.fixedname('sd', contrast, which=which, gzipped=gzipped, resampled=True)))
            df.append(100.)
        except:
            Warning('subject:%d does not have data for contrast:%s, eventtype:%s' % (subject.subj, contrast, which))
            pass

    fwhmraw = FWHMestimator(eff_files[0], fwhm='fwhm-%s-%s.img' % (contrast, which), resels='resels-%s.img' % contrast)
    subjects[0].resample(os.path.join(subjects[0].fsldir, 'fsl1', 'mask.img'))
    mask = BrainSTAT.VImage(os.path.join(subjects[0].fsldir, 'fsl1', 'mask_rsmpl.img'))
    multi = MultiStat(eff_files, 'multi-%s-%s' % (contrast, which),
                      sd_files=sd_files,
                      fwhmfile='fwhm-OLS-%s-%s.img' % (contrast, which),
                      reselsfile='resels-OLS-%s-%s.img' % (contrast, which),
                      clobber=True,
                      df=df,
                      fwhmmask=mask)


across_subjects(subjects, 'sentence', which='block')
across_subjects(subjects, 'sentence', which='event')

def localFDRfit(histogram, trim=0.1, scale=1):
    n = histogram.shape[1]
    Q = cumsum(histogram[1])
    start = min(arange(n)[greater(Q, trim)])
    end = max(arange(n)[less(Q, 1.-trim)])
    _histogram = histogram[:,start:end]
    binssq = _histogram[0]**2
    design = transpose(array([ones(_histogram.shape[1:], Float), _histogram[0], binssq]))
    model = GLM.GLM(_histogram[1]*scale, design, family=GLM.Poisson())
    model.fit()

    coef = model.lm.beta
    sigma = 1 / sqrt(-2 * coef[2])
    mu = coef[1] * sigma**2

    print sigma, mu 
    def _approx(x):
        return exp(coef[0] + coef[1]*x + coef[2] * x**2) / scale

    def _capprox(x):
        return exp(coef[0] + coef[1]*(x+mu) + coef[2] * (x+mu)**2) / scale

    pi0 = sum(_approx(_histogram[0])) / sum(_histogram[1])

    return sigma, mu, pi0, _approx, _capprox

def maskedT(subj, contrast, which='block'):
    subject = subjects[subj]
    results = subject.results[which][contrast]
    ntrial = len(results)
    if ntrial == 1:
        mask = subject.mask(run=subject.type[which][0])
    else:
        mask1 = subject.mask(run=subject.type[which][0])
        _mask1 = mask1.readall()

        mask2 = subject.mask(run=subject.type[which][1])
        _mask2 = mask2.readall()

        mask = BrainSTAT.VImage(_mask1 * _mask2, dimensions=mask1.dimensions)
        
    tfile = BrainSTAT.VImage(fixedname('t', subj, contrast, which=which))
    return tfile, mask

def histogram(timage, bins=arange(-7,7,0.1), mask=None, plot=False, name=None):

    histogram = Histogram(timage, bins, mask=mask)
    N = sum(histogram[1])
    histogram[1] = histogram[1] / N
    df = 180
    tP = gslutils.tdist_P(bins, df)
    tP = tP[1:] - tP[:-1]

    sigma, mu, pi0, locFDR, locFDRc = localFDRfit(histogram, scale=N)

    if plot:
        pylab.plot(bins[:-1], tP, linewidth=2, color='red')
        pylab.bar(histogram[0], histogram[1], width=0.1)
        pylab.plot(bins, locFDR(bins), linewidth=2, color='yellow')
        pylab.plot(bins, locFDRc(bins), linewidth=2, color='green')
        if name:
            pylab.title('Histogram of %s' % name)
            pylab.savefig(name[0:-4] + '-hist.png')
        pylab.show()
        pylab.clf()
    

    empirical = interpolant(histogram[0], histogram[1])
    theoretical = interpolant(bins[:-1], tP)
    cempirical = interpolant(histogram[0], histogram[1])

    return theoretical, empirical, locFDR, locFDRc, cempirical

def getcov(subj, contrast1, contrast2, which='block', run=1):
    subject = subjects[subj]
    if subject.validate(run=run):
        try:
            _filename = os.path.join(subject.fsldir, 'fsl%d' % run, 'filtered_func_data_cov_%s_%s.img' % (contrast1, contrast2))
            _cov = BrainSTAT.VImage(_filename).readall()
        except:
            _filename = os.path.join(subject.fsldir, 'fsl%d' % run, 'filtered_func_data_cov_%s_%s.img' % (contrast2, contrast1))
            _cov = BrainSTAT.VImage(_filename).readall()
        return _cov

def geteff(subj, contrast, which='block', run=1):
    subject = subjects[subj]
    if subject.validate(run=run):
        _filename = os.path.join(subject.fsldir, 'fsl%d' % run, 'filtered_func_data_%s_eff.img' % contrast)
        _eff = BrainSTAT.VImage(_filename).readall()
        return _eff

def gett(subj, contrast, which='block', run=1):
    subject = subjects[subj]
    if subject.validate(run=run):
        _filename = os.path.join(subject.fsldir, 'fsl%d' % run, 'filtered_func_data_%s_t.img' % contrast)
        _t = BrainSTAT.VImage(_filename).readall()
        return _t

def samplecontrast(subj, which='block', run=1, contrast=None):
    '''Generate a random contrast, by default selected perpendicular to "overall" activity. '''

    _keys = FIAC.contrasts.keys()
    _keys.pop(_keys.index('overall'))
    if contrast is None:
        contrast = standard_normal((3,))
        contrast = contrast - mean(contrast)

    # form effect

    #    print contrast

    _eff = 0.
    _var = 0.
    __var = []
    for i in range(3):
        Ikey = _keys[i]
        __var.append(getcov(subj, Ikey, Ikey, which=which, run=run))
        _eff = _eff + gett(subj, Ikey, which=which, run=run) * contrast[i]
        _var = _var + contrast[i]**2
        for j in range(i):
             Jkey = _keys[j]
             _cov = getcov(subj, Ikey, Jkey, which=which, run=run)
             _var = _var + contrast[i] * contrast[j] * _cov * inv(sqrt(__var[i] * __var[j]))

    template = BrainSTAT.VImage(fixedname('t', 0, 'overall'))

    print add.reduce(less_equal(_var, 0).flat.astype(Int))
    _sd = sqrt(_var)
    _t = _eff * inv(_sd)

    eff = BrainSTAT.VImage(_eff, dimensions=template.dimensions)
    sd = BrainSTAT.VImage(_sd, dimensions=template.dimensions) 
    t = BrainSTAT.VImage(_t, dimensions=template.dimensions) 

    return eff, sd, t

import time

eff, sd, t =  samplecontrast(0)
theoretical, empirical, localFDR, localFDRc, cempirical = histogram(t, plot=True)

toc = time.time()
for i in range(1000):
    try:
        eff, sd, t =  samplecontrast(random_integers(-1,16), run=random_integers(1,5), which='event')
        theoretical, empirical, localFDR, localFDRc, cempirical = histogram(t, plot=False)
    except:
        pass
tic = time.time()
print 'Time:', `tic-toc`

fixedeffects = False
dohistogram = True

for subj in range(16):
    for contrast in FIAC.contrasts.keys():
        for exptype in ['block', 'event']:
            for extra in ['', '_delay']:
                if fixedeffects:
                    print 'Fixed effect analysis: subject=%d; contrast=%s%s; exptype=%s' % (subj, contrast, extra, exptype)
                    within_subject(subj, '%s%s' % (contrast, extra), which=exptype)
                if dohistogram:
                    print 'Histogram: subject=%d; contrast=%s%s; exptype=%s' % (subj, contrast, extra, exptype)
                    tfile, mask = maskedT(subj, '%s%s' % (contrast, extra), which=exptype)
                    histogram(tfile, mask=mask, plot=True, name=tfile.image.filename)
                    
