import os, glob, BrainSTAT, csv, string, types, matplotlib, pylab, gc, shutil
from BrainSTAT.Base.Dimension import Dimension
from BrainSTAT.fMRIstat.PCA import *
from BrainSTAT.fMRIstat.SingleSubject import *
from BrainSTAT.fMRIstat.MultiStat import *
from BrainSTAT.fMRIstat import Design, HRF
from numarray import *
from BrainSTAT.Modules.KernelSmooth import LinearFilter3d
import BrainSTAT.Visualization.Pylab as Pylab
from BrainSTAT.fMRIstat.FWHMestimator import FWHMestimator
from ftplib import FTP
from optparse import OptionParser
from BrainSTAT.Base import Warp
from BrainSTAT.Utils.miscutils import reslicer
import ftplib, tarfile

## merely for readability of the options
## table, gets written over later with actual images

refs = ['average', 'standard']

## merely for readability of the options
## table, these are really defined in the class FIAC below

contrasts = ['sentence', 'overall', 'speaker', 'interaction']

## merely for readability of the options
## table, these are really defined in the class FIAC below

types = ['event', 'block']
interesting = (31,51,17)
parser = OptionParser(usage="%prog [options] url")

parser.add_option('', '--basedir', dest='basedir', help='base directory for the study', default='/home/analysis/FIAC')
parser.add_option('', '--getdata', dest='getdata', action='store_true', help='download the data?', default=False)
parser.add_option('', '--make4d', dest='make4d', action='store_true', help='make 4d files from 3d files?', default=False)
parser.add_option('', '--fsl', dest='fsl', action='store_true', help='run fsl script?', default=False)
parser.add_option('', '--usenonfsl', dest='usenonfsl', action='store_true', help='run fsl script?', default=False)
parser.add_option('', '--pca', dest='pca', action='store_true', help='run PCA analysis?', default=False)
parser.add_option('', '--mask', dest='mask', action='store_true', help='plot mask', default=False)
parser.add_option('', '--design', dest='design', action='store_true', help='plot design?', default=False)
parser.add_option('', '--runs', dest='runs', help='comma separated list of runs from 1 to 4, e.g. "--runs=1,2,3"', default='1,2,3,4')
parser.add_option('', '--singlerun', dest='singlerun', action='store_true', help='run single run model?', default=False)
parser.add_option('', '--fwhm', dest='fwhm', help='FWHM output file in singlerun', default='')
parser.add_option('', '--resels', dest='resels', help='resels output file in singlerun', default='')
parser.add_option('', '--fixed', dest='fixed', action='store_true', help='do fixed effects analysis', default=False)
parser.add_option('', '--resample', dest='resample', action='store_true', help='force resampling of effects, sd for within subjects analysis?', default=False)
parser.add_option('', '--multidir', dest='multidir', help='directory to store multistat results', default='./')
parser.add_option('', '--multistat', dest='multistat', help='run multistat analysis', action='store_true', default=False)
parser.add_option('', '--multiall', dest='multiall', help='run all multistat analyses?', action='store_true', default=False)
parser.add_option('', '--plot', dest='plot', help='plot results of multistat analysis', action='store_true', default=False)
parser.add_option('', '--plotslice', dest='plotslice', help='which slice to plot', default=interesting[0], type='int')
parser.add_option('', '--plotaxis', dest='plotaxis', help='which axis is plotslice defined on?', default=0, type='int')
parser.add_option('', '--view', dest='view', help='view results of multistat analysis', action='store_true', default=False)
parser.add_option('', '--slices', dest='slices', help='if viewing, which slices should we open the viewer at?', default=interesting)
parser.add_option('', '--ref', dest='ref', help='if viewing, what anatomical reference: one of "%s"' % string.join(refs, ','), default='average')
parser.add_option('', '--eventtype', dest='which', help='if viewing, which event type, one of "%s" ' % string.join(types, ','), default='block')
parser.add_option('', '--delay', dest='delay', help='if viewing, view delay or magnitude?', default=False, action='store_true')
parser.add_option('', '--contrast', dest='contrast', help='if viewing, which contrast of interest? one of "%s"' % string.join(contrasts, ','), default='overall')
parser.add_option('', '--stat', dest='stat', help='if viewing, which statistic should we view, one of "eff,sd,t"', default='t')
parser.add_option('', '--thresh', dest='thresh', help='if viewing, what threshold should we use for the image?', default=4.0, type='float')
parser.add_option('', '--view3d', dest='view3d', help='view 3dresults of multistat analysis', action='store_true', default=False)
parser.add_option('', '--verbose', dest='verbose', help='verbose output', action='store_true', default=False)

try:
    options, args = parser.parse_args()
except:
    parser.print_help()
    sys.exit()

def plot_3dimage(image, pngfile=None, mask_data=None, show=False, cmap=Pylab.spectral, **keywords):
    '''Create a 5x6 plot of slices for given 3d image.'''
    _image = BrainSTAT.VImage(image)
    data = _image.toarray().image.data
    normalize = matplotlib.colors.normalize()
##     cax = pylab.axes([0.87, 0.1, 0.03, 0.38])
##     pylab.colorbar(cax=cax) # how can I make the colormap consistent across plots?

    data = Pylab.spectral(normalize(data), **keywords)

    if mask_data is not None:
        for i in range(3):
            data[:,:,:,i] = where(mask_data, data[:,:,:,i], 0.)
    
    for i in range(5):
        for j in range(6):
            Pylab.pylab.set(Pylab.pylab.gca(), 'yticklabels', [])
            Pylab.pylab.set(Pylab.pylab.gca(), 'xticklabels', [])
            Pylab.pylab.subplot(5,7,i*6+j+1)
            if mask_data is None:
                Pylab.pylab.imshow(data[6*i+j], cmap=cmap, interpolation='bicubic', vmax=1.0, vmin=0.0, **keywords)
            else:
                Pylab.pylab.imshow(data[6*i+j], interpolation='bicubic', vmax=1.0, vmin=0.0, **keywords)
    if pngfile:
        Pylab.pylab.savefig(pngfile)
    if show:
        Pylab.pylab.show()
    Pylab.pylab.clf()

def _make4d(input_files, output_file, fwhm=0.0):
    '''Take the list of (assumed identical) input_files and output a concatenated version in output_file.'''
    images = map(lambda x: BrainSTAT.VImage(x).toarray(), input_files)

    timedim = Dimension('time', len(input_files), start=0.0, step=2.5)

    template = images[0]

    output = BrainSTAT.fMRIImage(output_file, mode='w', dimensions=(timedim,) + template.dimensions, template=template, clobber=True)

    if fwhm > 0.0:
        kernel = LinearFilter3d(template, norm=1.0, fwhm=fwhm)

    for i in range(len(input_files)):
        if fwhm > 0.0:
            cur_data = kernel.smooth(images[i])
        else:
            cur_data = images[i]
        output.image.write((i,0,0,0), array([cur_data.image.data]))

class FIAC:

    basedir = options.basedir

    ftphost = 'ftp.cea.fr'
    ftpdir = '/pub/dsv/madic/FIAC'
    anonemail = 'jonathan.taylor@stanford.edu'

    ncomp = 6

    contrasts = {}
    contrasts['overall'] = {'SSt_SSp':0.25, 'SSt_DSp':0.25, 'DSt_SSp':0.25, 'DSt_DSp':0.25}
    contrasts['sentence'] = {'SSt_SSp':-0.5, 'SSt_DSp':-0.5, 'DSt_SSp':0.5, 'DSt_DSp':0.5}
    contrasts['speaker'] = {'SSt_SSp':-0.5, 'SSt_DSp':0.5, 'DSt_SSp':-0.5, 'DSt_DSp':0.5}
    contrasts['interaction'] = {'SSt_SSp':1.0, 'SSt_DSp':-1.0, 'DSt_SSp':-1.0, 'DSt_DSp':1.0}
    
    types = ['block' ,'event']

    mapping = ['SSt_SSp', 'SSt_DSp', 'DSt_SSp', 'DSt_DSp']

    runs = range(1,5)
    
    first_frame = 5

    def __init__(self, subj):
        self.subj = subj
        self.dir = os.path.join(FIAC.basedir, 'fiac%d' % self.subj)
        self.fsldir = os.path.join('/usr/local/FIAC/', 'fiac%d' % self.subj)
        if not os.path.exists(os.path.join(self.fsldir)):
            os.mkdir(self.fsldir)

        self.type = {}
        self.df = {}
        for type in FIAC.types:
            self.type[type] = []

        self.valid_runs = []
        for i in FIAC.runs:
            if self.validate(run=i):
                self.valid_runs.append(i)

    def getlisting(self):
        '''Get list of files in subject directory on FIAC website.'''
        self.ftpserver = ftplib.FTP(FIAC.ftphost)
        self.ftpserver.login('anonymous', FIAC.anonemail)
        return self.ftpserver.nlst(os.path.join(FIAC.ftpdir, 'fiac%d' % self.subj))

    def rundir(self, run=runs[0]):
        return os.path.join(self.dir, 'fonc%d' % run)

    def validate(self, run=runs[0], frame=first_frame, which=None, verbose=False):
        if which is None:
            filename = os.path.join(self.rundir(run), 'fiac%d_fonc%d_%04d.img' % (self.subj, run, frame))
        if which is 'pca':
            filename = os.path.join(self.dir, 'pca', 'fiac%d_fonc%d_pca.csv' % (self.subj, run))
        elif which is 'design':
            filename = os.path.join(self.dir, 'design', 'fiac%d_fonc%d-design.png' % (self.subj, run))
        if verbose:
            print 'Validating filename: %s' % filename
        return os.path.exists(filename)


    def getdata(self, get_tar=True):
        '''Download all files for given subject from FIAC website.'''
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        os.chdir(FIAC.basedir)

        for ftpfile in self.getlisting():
            if os.path.splitext(ftpfile)[1] in ['.tar', '.gz']:
                if get_tar:
                    outfile = file(os.path.split(ftpfile)[1], 'wb')
                    self.ftpserver.retrbinary('RETR %s' % ftpfile, outfile.write)
                    outfile.close()

                    if os.path.splitext(outfile.name)[1] == '.tar':
                        _tarfile = tarfile.open(outfile.name, 'r')
                    else:
                        _tarfile = tarfile.open(outfile.name, 'r:gz')
                    for _file in _tarfile:
                        if _file.isdir():
                            if not os.path.exists(_file.name):
                                os.mkdir(_file.name)
                        else:
                            _tarfile.extract(_file.name, self.dir)
                    _tarfile.close()
                    os.remove(_tarfile.name)
            else:
                outfile = file(os.path.split(ftpfile)[1], 'wb')
                self.ftpserver.retrbinary('RETR %s' % ftpfile, outfile.write)
                outfile.close()
                os.rename(outfile.name, os.path.join('fiac%d' % self.subj, outfile.name))

    def write_fsl(self, run):
        if not os.path.exists(os.path.join(self.fsldir, 'fsl%d' % run)):
            os.mkdir(os.path.join(self.fsldir, 'fsl%d' % run))
            
        design = file(os.path.join(FIAC.basedir, 'fsl', 'design.fsf.template')).read()
        design = design.replace('ANAT', os.path.join(self.dir, 'anat1', 'fiac%d_anat1_brain' % self.subj))
        design = design.replace('OUTDIR', os.path.join(self.fsldir, 'fsl%d' % run))
        design = design.replace('FONC', os.path.join(self.dir, 'fonc%d' % run, 'fiac%d_fonc%d' % (self.subj, run)))

        designfile = file(os.path.join(self.fsldir, 'fsl%d' % run, 'design.fsf'), 'w')
        designfile.write(design)
        designfile.close()

    def execute_fsl(self, run):
        os.chdir(os.path.join(self.fsldir, 'fsl%d' % run))
        os.system('%s %d %d &> fsl.log' % (os.path.join(FIAC.basedir, 'fsl', 'fiac.bash'), self.subj, run))

    def make4d(self, fwhm=0.0, run=1):
        '''Make a 4-dimensional file from all the 3-dimensional files.'''
        if self.validate(run=run):
            input_files = []
            for frame in range(5, 196):
                input_files.append(os.path.join(self.dir, 'fonc%(run)d/fiac%(subj)d_fonc%(run)d_%(frame)04d.img' % {'subj':self.subj, 'run':run, 'frame':frame}))
            _make4d(input_files, os.path.join(self.dir, 'fiac%(subj)d_fonc%(run)d.img' % {'subj':self.subj, 'run':run}), fwhm=fwhm)

    def anat(self, nonfsl=options.usenonfsl):
        transform = BrainSTAT.Warp.fromfile(file(os.path.join(self.dir, 'fiac%d_anat1.txt' % self.subj)))
        if nonfsl:
            filename = os.path.join(self.dir, 'anat1', 'fiac%d_anat1.img' % self.subj)
        else:
            filename = os.path.join(self.dir, 'anat1', 'fiac%d_anat1_brain.img' % self.subj)

        _anat = BrainSTAT.VImage(filename)
        _anat.warp.translate_output(x=[-1.421875*63.5, -125.5, -0.898475*125.5])
        return _anat
    
    def fMRI(self, run=1, nonfsl=options.usenonfsl):
        '''Return the fMRIImage 4d file for a given subject and run.'''
        if self.validate(run=run):
            if nonfsl:
                image = BrainSTAT.fMRIImage(os.path.join(self.dir, 'fiac%d_fonc%d.img' % (self.subj, run)))
            else:
                image = BrainSTAT.fMRIImage(os.path.join(self.fsldir, 'fsl%d' % run, 'filtered_func_data.img'))
            return image

    def frame(self, frame=first_frame, run=1):
        '''Return the VImage 3d file for a given subject, frame and run.'''
        if self.validate(run=run, frame=frame):
            return BrainSTAT.VImage(os.path.join(self.dir, 'fonc%d' % run, 'fiac%d_fonc%d_%04d.img' % (self.subj, run, frame)))

    def pca(self, run=1, thresh=1800.):
        '''Perform a PCA analysis on a given subject\'s run, first FIAC.ncomp components are output. No variables regressed out.'''
        fMRI = self.fMRI(run=run)

        mask = self.mask(run=run, thresh=thresh)

        V, pcntvar, output = PCA(fMRI, which=range(FIAC.ncomp), clobber=True, verbose=True, mask=mask, output=True, mask_test=None)

        pcaprefix = os.path.join(self.dir, 'pca')
        if not os.path.exists(pcaprefix):
            os.mkdir(pcaprefix)
        for i in range(len(output)):
            outfile = os.path.join(pcaprefix, 'fiac%d_fonc%d_pca-comp%d.img' % (self.subj, run, i))
            output[i].tofile(outfile, clobber=True)

        pcafile = file(os.path.join(pcaprefix, 'fiac%d_fonc%d_pca.csv' % (self.subj, run)), 'w')
        writer = csv.writer(pcafile)
        writer.writerows(V)
        pcafile.close()
        del(fMRI)
        del(output)
        gc.collect()
        
    def plot_pca_timecourse(self, run=1):
        '''Basic plot of the time series part of the PCA analysis. Plots all components: lowest corresponding to 1st component.'''
        if self.validate(run=run, which='pca'):
            pngfile = os.path.join(self.dir, 'pca', 'fiac%d_fonc%d_pca-time.png' % (self.subj, run))
            pcafile = file(os.path.join(self.dir, 'pca', 'fiac%d_fonc%d_pca.csv' % (self.subj, run)))
            V = []
            for row in csv.reader(pcafile):
                V.append(map(string.atof, row))
            V = transpose(array(V))
            Pylab.multipleSeriesPlot(V)
            Pylab.pylab.savefig(pngfile)
            Pylab.pylab.clf()

    def plot_pca_image(self, run=1, comp=0, mask_data=None, **keywords):
        '''Create a 5x6 grid plot of slices for given run and PCA component.'''

        if self.validate(run=run, which='pca'):
            pngfile = os.path.join(self.dir, 'pca', 'fiac%d_fonc%d_pca-comp%d-space.png' % (self.subj, run, comp))
            plot_3dimage(os.path.join(self.dir, 'pca', 'fiac%d_fonc%d_pca-comp%d.img' % (self.subj, run, comp)), pngfile=pngfile, mask_data = mask_data, **keywords)

    def mask(self, run=1, thresh=1800., show=True, nonfsl=options.usenonfsl):
        '''Return a mask image for each run, subject. Defaults to FSL mask.'''

        if self.validate(run=run):
            pngfile = os.path.join(self.dir, 'fiac%d_fonc%d_mask.png' % (self.subj, run))
            if nonfsl:
                _mask = BrainSTAT.VImage(greater(self.frame(frame=5, run=run).toarray().image.data, thresh))
                print 'Mask threshold: %0.2f' % thresh
            else:
                _mask = BrainSTAT.VImage(os.path.join(self.fsldir, 'fsl%d' % run, 'mask.img'))
            
            plot_3dimage(_mask, pngfile=pngfile, cmap=pylab.cm.gray)
            return _mask

    def design(self, run=1, eps=0.02, plot=False):
        '''Generate the design for a given subject, run.'''

        if not os.path.exists(os.path.join(self.dir, 'design')):
            os.mkdir(os.path.join(self.dir, 'design'))
        if self.validate(run=run):
            IRF = HRF.spectralHRF(ncomp=2)
            eventfile = glob.glob(os.path.join(self.dir, 'subj*fonc%d*txt' % run))[0]
            if eventfile.find('bloc') >= 0:
                self.type['block'].append(run)
                _type = 'block'
            else:
                self.type['event'].append(run)
                _type = 'event'

            eventfile = file(eventfile)
            times = []
            events = {}
            for name in FIAC.mapping:
                events[name] = []

            for row in eventfile:
                time, eventtype = row.split()
                time = string.atof(time)
                times.append(time)
                eventtype = string.atoi(eventtype)
                name = FIAC.mapping[eventtype-1]
                events[name].append(Design.Event(start = time, duration = 10./3 - eps, height=1.0, name=name, IRF=IRF))
            time = array(times)

            stimuli = {}
            for i in range(4):
                name = FIAC.mapping[i]
                stimuli[name] = Design.Events2Stimuli(events[name])[0]

            fMRI = self.fMRI(run=run)

            _events = []
            for name in events.keys():
                _events = _events + events[name]

            design = Design.Design(fMRI.time.values(), _events)
            design._premodel()

            self.model = design.model(fMRI.time.values())

            if plot:
                pngfile = os.path.join(self.dir, 'design', 'fiac%d_fonc%d_design.png' % (self.subj, run))
                design.plot()
                Pylab.pylab.savefig(pngfile)
                Pylab.pylab.clf()
            
            del(fMRI)
            gc.collect()
            self._design = design
            return design

    def extractors(self, design, singlerun, run=1, fwhm=options.fwhm, resels=options.fwhm):
        '''Setup extractors for singlerun analysis -- probably should make a separate directory within the FSL directory -- oh well...'''
        for event in FIAC.mapping:
            singlerun.extractors = singlerun.extractors + DelayExtractor(design, singlerun.image, event)

        for effect in FIAC.contrasts.keys():
            singlerun.Tcontrast(effect, **FIAC.contrasts[effect])
            singlerun.extractors = singlerun.extractors + DelayContrast(singlerun.image, effect, **FIAC.contrasts[effect])

        def getrho(results):
            return results.rho

        npixel = product(singlerun.image.spatial_shape[1:])
        rho_extractor = Extractor('rho', getrho)
        rho_extractor.setup((npixel,), iter(BrainSTAT.VImage('%s_rho.img' % singlerun.image.image.filebase, mode='w', dimensions=singlerun.image.spatial_dimensions)))
        singlerun.extractors.append(rho_extractor)

        def getvar(results):
            return results.var

        npixel = product(singlerun.image.spatial_shape[1:])
        var_extractor = Extractor('var', getvar)
        var_extractor.setup((npixel,), iter(BrainSTAT.VImage('%s_var.img' % singlerun.image.image.filebase, mode='w', dimensions=singlerun.image.spatial_dimensions)))
        singlerun.extractors.append(var_extractor)

        if resels or fwhm:

            self.ref_frame = self.fMRI(run=run).toarray(slice=(95,))

            def extract_wresid(results):
                value = transpose(results.norm_resid)
                return value

            self.fwhmest = FWHMestimator(self.ref_frame, fwhm=fwhm, resels=resels, mask=self.mask(run=run))
            fwhm_extract = Extractor('fwhm', extract_wresid)
            fwhm_extract.setup((npixel, singlerun.nkeep), iter(self.fwhmest))
            
            singlerun.extractors.append(fwhm_extract)

##         for i in range(4):
##             for j in range(i+1):
##                 _nameI = FIAC.contrasts.keys()[i]
##                 _nameJ = FIAC.contrasts.keys()[j]

##                 singlerun.extractors.append(ContrastCovarianceExtractor(self.design, self.fMRI(run=run), _nameI, FIAC.contrasts[_nameI], _nameJ, FIAC.contrasts[_nameJ]))
                    
    def _results(self, result, run=1, resampled=False):
        '''Return (eff, sd, t) maps for "result" in a given "run".'''
        if resampled:
            _rsmpl = '_rsmpl'
        else:
            _rsmpl = ''
        if self.validate(run=run):
            _dir = os.path.join(self.fsldir, 'fsl%d' % run)
            _sd  = BrainSTAT.VImage(os.path.join(_dir, 'filtered_func_data_%s_sd%s.img' % (result, _rsmpl)))
            _eff  = BrainSTAT.VImage(os.path.join(_dir, 'filtered_func_data_%s_eff%s.img' % (result, _rsmpl)))
            _t  = BrainSTAT.VImage(os.path.join(_dir, 'filtered_func_data_%s_t%s.img' % (result, _rsmpl)))
            return (_eff, _sd, _t)
        
    def gettype(self):
        '''Determine which runs are block, which are events.'''
        self.type = {}
        self.type['block'] = []
        self.type['event'] = []
        for run in FIAC.runs:
            if self.validate(run=run):
                eventfile = glob.glob(os.path.join(self.dir, 'subj*fonc%d*txt' % run))[0]
                if eventfile.find('bloc') >= 0:
                    self.type['block'].append(run)
                else:
                    self.type['event'].append(run)

    def getresults(self, resampled=False):
        '''Fill in the attribute "results" with the output of each event type and each contrast of interest.'''

        self.results = {}
        self.gettype()
        for which in ['block', 'event']:
            _results = {}
            for _contrast in FIAC.contrasts.keys():
                for extra in ['', '_delay']:
                    __contrast = '%s%s' % (_contrast, extra)
                    _results[__contrast] = []
                    for run in self.type[which]:
                        if resampled:
                            __eff, __sd, __t = self._results(__contrast, run=run, resampled=False)
                            _eff = self.resample(__eff.image.filename, run=run, force=options.resample)
                            _sd = self.resample(__sd.image.filename, run=run, force=options.resample)
                            _t = self.resample(__t.image.filename, run=run, force=options.resample)
                            _value = (_eff, _sd, _t)
                        else:
                            _value = self._results(__contrast, run=run)
                        if _value:
                            _results[__contrast].append(_value)
            self.results[which] = _results

    def getdf(self):
        self.gettype()
        for which in self.type.keys():
            self.df[which] = len(self.type[which]) * 100.

    def pca_webpage(self):
        '''Create a summary webpage of the PCA data for each subject.'''
        webfile = file(os.path.join(self.dir, 'pca.html'), 'w')
        webpage = file(os.path.join(FIAC.basedir, 'python', 'index.html.template')).read()

        title = 'PCA summary for subject %d' % self.subj
        webpage = webpage.replace('TITLE', title)

        body = '<h1>PCA in time:</h1>\n'
        for run in FIAC.runs:
            if self.validate(run=run, which='pca'):
                pngfile = 'pca/fiac%d_fonc%d_pca-time.png' % (self.subj, run)
                body = body + '<h2>Run %d</h2>\n<img src="%s">\n' % (run, pngfile)

        body = body + '<h1>PCA in space:</h1>\n'
        for run in FIAC.runs:
            if self.validate(run=run, which='pca'):
                body = body + '<h2>Run %d</h2>\n' % run
                pngfile = 'fiac%d_fonc%d_mask.png' % (self.subj, run)
                body = body + '<h2>Mask image used for PCA</h2>\n<img src="%s">\n' % (pngfile)
                for comp in range(FIAC.ncomp):
                    pngfile = 'pca/fiac%d_fonc%d_pca-comp%d-space.png' % (self.subj, run, comp)
                    body = body + '<h3>Run %d, Component %d</h3>\n<img src="%s">\n' % (run, comp + 1, pngfile)
            
        webpage = webpage.replace('BODY', body)
        webfile.write(webpage)
        webfile.close()

    def design_webpage(self):
        '''Create a summary webpage of the design for each subject.'''
        webfile = file(os.path.join(self.dir, 'design.html'), 'w')
        webpage = file(os.path.join(FIAC.basedir, 'python', 'index.html.template')).read()

        title = 'Design summary for subject %d' % self.subj
        webpage = webpage.replace('TITLE', title)

        body = ''
        for run in FIAC.runs:
            if self.validate(run=run):
                pngfile = 'fiac%d_fonc%d_design.png' % (self.subj, run)
                body = body + '<h2>Run %d</h2>\n<img src="%s">\n' % (run, pngfile)
            
        webpage = webpage.replace('BODY', body)
        webfile.write(webpage)
        webfile.close()
    
    gzipped = False

    def fixedname(self, stat, contrast, which='block', gzipped=gzipped, resampled=False):
        '''For a given statistic type, contrast and event type, return the name of the corresponding fixed effect file.'''

        self.fixeddir = os.path.join(self.fsldir, 'fixed')

        if not os.path.exists(self.fixeddir):
            os.mkdir(self.fixeddir)
            os.system('ln -s %s %s' % (self.fixeddir, '/home/analysis/FIAC/fiac%d/fixed' % subj))

        if gzipped:
            _gzip = '.gz'
        else:
            _gzip = ''

        if resampled:
            _rsmpl = '_rsmpl'
        else:
            _rsmpl = ''

        return os.path.join(self.fixeddir, '%s_%s_%s%s.img%s' % (contrast, which, stat, _rsmpl, _gzip))

    def within_subject(self, contrast, which='block', gzipped=gzipped):
        '''For a given contrast and event type, perform a within-subject fixed effects analysis -- basically just weight the effect images inversely proportional to the separate SD files. If there is just one run, then do the sensible thing: output the results of this run.'''
        results = self.results[which][contrast]

        ntrial = len(results)

        print 'Within subjects: %s,%s' % (contrast, which)
        if ntrial > 1:
            eff1, sd1, t1 = results[0]
            eff2, sd2, t2 = results[1]
            
            eff_files = [eff1, eff2]
            sd_files = [sd1, sd2]

            multi = MultiStat(eff_files, os.path.join(self.dir, 'fixed', '%s_%s' % (contrast, which)),
                              sd_files=sd_files,
                              verbose=False,
                              clobber=True,
                              df=[100] * ntrial,
                              fixed=True)
        else:
            eff, sd, t = results[0]
            if eff is not None:
                shutil.copyfile(eff.image.filename, os.path.join(self.dir, 'fixed', '%s_%s_effect.img' % (contrast, which)))
                shutil.copyfile(eff.image.filename[0:-3] + 'hdr', os.path.join(self.dir, 'fixed', '%s_%s_effect.hdr' % (contrast, which)))
                shutil.copyfile(sd.image.filename, os.path.join(self.dir, 'fixed', '%s_%s_sd.img' % (contrast, which)))
                shutil.copyfile(sd.image.filename[0:-3] + 'hdr', os.path.join(self.dir, 'fixed', '%s_%s_sd.hdr' % (contrast, which)))

            
    def resample(self, filename, run=runs[0], force=True, verbose=False, flirt=False):
        if self.validate(run=run):
            output = os.path.abspath(os.path.splitext(filename)[0] + '_rsmpl.img')
            if not os.path.exists(output) or force:
                if flirt:
                    cmd = "flirt -ref %(standard)s -in %(input)s -out %(output)s -applyxfm -init %(mat)s -interp trilinear" % {'standard':'/home/stow/fsl/etc/standard/avg152T1_brain.img',
                                                                                                                               'input': os.path.abspath(os.path.splitext(filename)[0]),
                                                                                                                               'output': os.path.abspath(os.path.splitext(filename)[0] + '_rsmpl'),
                                                                                                                               'mat': os.path.join(self.fsldir, 'fsl%d' % run, 'example_func2standard.mat')}
                    if verbose:
                        print cmd
                    os.system(cmd)
                else:
                    try:
                        transform = Warp.fromfile(file(os.path.join(self.fsldir, 'fsl%d' % run, 'example_func2standard.xfm')))
                    except:
                        Warning.Warning('cannot find file %s, no resampling done' % os.path.join(self.fsldir, 'fsl%d' % run, 'example_func2standard.xfm'))
                        return None

                    inputim = BrainSTAT.VImage(filename)
                    warp = Warp.Affine(inputim.warp.input_coords, refs['standard'].warp.input_coords, transform)
                    warp.reorder(reorder_dims=False)
                    outputim = inputim.resample(warp, input='voxel')
                    outputim.dimensions = average.dimensions
                    outputim.tofile(output)
            return BrainSTAT.VImage(output)

def across_subjects(subjects, contrast='sentence', which='block', multidir=os.path.join(FIAC.basedir, 'multi'), verbose=True, keith=False):
    eff_files = []
    sd_files = []
    df = []

    if not os.path.exists(multidir):
        os.mkdir(multidir)

    os.chdir(multidir)

    if keith:
        for subject in good_subjects:
            try:
                eff_files.append(BrainSTAT.VImage('http://www.math.mcgill.ca/keith/jonathan/subj%d_bloc_sen_del_ef.img' % subject, urlstrip='/keith/jonathan/'))
                sd_files.append(BrainSTAT.VImage('http://www.math.mcgill.ca/keith/jonathan/subj%d_bloc_sen_del_sd.img' % subject, urlstrip='/keith/jonathan/'))
                df.append(100.)
            except:            
                Warning.Warning('subject:%d does not have data for contrast:%s, eventtype:%s' % (subject, contrast, which))
                pass
    else:
        for subject in subjects:
            subject.getdf()
            try:
                eff_files.append(BrainSTAT.VImage(subject.fixedname('effect', contrast, which=which)))
                sd_files.append(BrainSTAT.VImage(subject.fixedname('sd', contrast, which=which)))
                df.append(100.) ##subject.df[which])
            except:            
                Warning.Warning('subject:%d does not have data for contrast:%s, eventtype:%s' % (subject.subj, contrast, which))
                pass

    fwhmraw = FWHMestimator(eff_files[0], fwhm='fwhm-%s-%s.img' % (contrast, which), resels='resels-%s-%s.img' % (contrast, which))
    multi = MultiStat(eff_files, 'multi-%s-%s' % (contrast, which),
                      sd_files=sd_files,
                      fwhmfile='fwhm-%s-%s.img' % (contrast, which),
                      reselsfile='resels-%s-%s.img' % (contrast, which),
                      verbose=False,
                      clobber=True,
                      df=df,
                      fwhmmask=mask,
                      sdratio=True)

def average_anatomy(subjects, force=False):
    if not os.path.exists(os.path.join(FIAC.basedir, 'avganat.img')) or force:
        val = 0.
        for subj in subjects:
            anat = BrainSTAT.VImage(os.path.join(subj.fsldir, 'fsl%d' % subj.valid_runs[0], 'highres2standard.img'))
            val = val + anat.readall() / len(subjects)

        average = BrainSTAT.VImage('/home/analysis/FIAC/avganat.img', dimensions=anat.dimensions, mode='w')
        average.image.write((0,)*3, val)
    else:
        average = BrainSTAT.VImage('/home/analysis/FIAC/avganat.img')
    return average

def group_mask(subjects, force=False):
    if not os.path.exists(os.path.join(FIAC.basedir, 'mask.img')) or force:
        val = 1.
        for subj in subjects:
            for run in FIAC.runs:
                if subj.validate(run=run):
                    mask = subj.resample(subj.mask(run=run).image.filename, run=run, force=force)
                    val = val * mask.readall() 
        mask = BrainSTAT.VImage('/home/analysis/FIAC/mask.img', dimensions=mask.dimensions, mode='w')
        mask.image.write((0,)*3, val)
    else:
        mask = BrainSTAT.VImage('/home/analysis/FIAC/mask.img')
    return mask

##############################################################################
#
#
# Anatomical reference images
#
#
##############################################################################

standard = BrainSTAT.VImage('http://www.fil.ion.ucl.ac.uk/~john/misc/images/avg152T1.mnc', urlstrip='/~john/misc/images/') # appears to be the same as the FSL standard
good_subjects = [0,1,3,4,6,7,8,9,10,11,12,13,14,15]
average = average_anatomy([FIAC(subj) for subj in good_subjects])
mask = group_mask([FIAC(subj) for subj in good_subjects])

refs = {}
refs['average'] = average
refs['standard'] = standard

##############################################################################
#
#
# View multistat results
#
#
##############################################################################


def view_results(contrast='sentence', delay=False, which='block', anat=average, stat='t', thresh=4.0, mask=mask):
    '''View results of multistat analysis for a given contrast, delay choice, event type, anatomy reference, statistic choice and threshold.'''
    anatd = anat.readall()

    if delay:
        extra = '_delay'
        
    else:
        extra = ''

    multi = BrainSTAT.VImage('http://kff.stanford.edu/FIAC/multi/multi-%s%s-%s_%s.img' % (contrast, extra, which, stat), urlstrip='/FIAC/multi/', realm='FIAC Website', username='keith', password='poincare') 
    multid = multi.readall()

    maskd = mask.readall()

    alphad = maskd * greater_equal(abs(multid), thresh).astype(Float)

    multilayer = BrainSTAT.Visualization.Pylab.DataLayer(multid, alpha=alphad)
    anatlayer = BrainSTAT.Visualization.Pylab.DataLayer(anatd, cmap=pylab.cm.gray, alpha = 1. - alphad)

    viewer = BrainSTAT.Visualization.Pylab.ViewLayers([multilayer, anatlayer], verbose=options.verbose)
    viewer.draw(array([90,108,90]) - array(options.slices), show=True)


##############################################################################
#
#
# Plot multistat results
#
#
##############################################################################


def plot_results(contrast='sentence', delay=False, which='block', anat=average, stat='t', thresh=4.0, mask=mask, plotslice=interesting[0], plotaxis=0):
    '''View results of multistat analysis for a given contrast, delay choice, event type, anatomy reference, statistic choice and threshold.'''
    anatd = anat.readall()

    if delay:
        extra = '_delay'
        
    else:
        extra = ''

    multi = BrainSTAT.VImage('http://kff.stanford.edu/FIAC/multi/multi-%s%s-%s_%s.img' % (contrast, extra, which, stat), urlstrip='/FIAC/multi/', realm='FIAC Website', username='keith', password='poincare') 
    multid = multi.readall()

    maskd = mask.readall()

    alphad = maskd * greater_equal(abs(multid), thresh).astype(Float)

    anatd = reslicer(anatd, plotslice, plotaxis)
    multid = reslicer(multid, plotslice, plotaxis)
    alphad = reslicer(alphad, plotslice, plotaxis)

    multilayer = BrainSTAT.Visualization.Pylab.DataLayer(multid, alpha=alphad)
    anatlayer = BrainSTAT.Visualization.Pylab.DataLayer(anatd, cmap=pylab.cm.gray, alpha = 1. - alphad)

    _slice = BrainSTAT.Visualization.Pylab.PlotLayers([multilayer, anatlayer])
    _slice.draw(show=True)

##############################################################################
#
#
# View 3d multistat results
#
#
##############################################################################


def view3d_results(contrast='sentence', delay=False, which='block', anat=average, stat='t', thresh=4.0, mask=mask, anatthresh=2000., anatcolor=(0.75,)*3, multicolor=(1.0,0.3,0.3), pos=(276,539,-320), viewup=(0.961,-0.0947,0.259)):
    '''View 3d isosurface results of multistat analysis for a given contrast, delay choice, event type, anatomy reference, statistic choice and threshold.'''

    if delay:
        extra = '_delay'
        
    else:
        extra = ''

    multi = BrainSTAT.VImage('http://kff.stanford.edu/FIAC/multi/multi-%s%s-%s_%s.img' % (contrast, extra, which, stat), urlstrip='/FIAC/multi/', realm='FIAC Website', username='keith', password='poincare') 

    viz = BrainSTAT.Visualization.initialize()
    anat.isosurface(thresh=anatthresh, viz=viz)
    anat._iso.SetOpacity(0.5)
    anat._iso.SetColor(anatcolor)
    anat._iso.renwin.update_view(pos[0], pos[1], pos[2], viewup[0], viewup[1], viewup[2])

    multi.isosurface(thresh=thresh, viz=viz)
    multi._iso.SetColor(multicolor)

    anat._iso.renwin.Render()
    return multi, anat

if __name__ == '__main__':
    import sys, string, time

    subjects = [FIAC(subj) for subj in map(string.atoi, args)]
    runs = [string.atoi(run) for run in string.split(options.runs, ',')]

    toc = time.time()
    for subject in subjects:
        if options.getdata:
            subject.getdata()
        for run in runs:
            print 'Subject %d, run %d' % (subject.subj, run)
            if subject.validate(run=run):

                if options.make4d:
                    subject.make4d(fwhm=0.0, run=run)

                if options.fsl:
                    subject.write_fsl(run=run)
                    subject.execute_fsl(run=run)

                _mask = subject.mask(run=run).readall()

                if options.pca:
                    subject.pca(run=run, thresh=1800.)
                    subject.plot_pca_timecourse(run=run)
                    for comp in range(FIAC.ncomp):
                        subject.plot_pca_image(run=run, comp=comp, mask_data=_mask)
                if options.design or options.singlerun:
                    design = subject.design(run=run)
                    
                if options.singlerun:
                    outdir = os.path.join(subject.fsldir, 'fsl%d' % run)
                    os.chdir(outdir)
                    fMRI = subject.fMRI(run=run)

                    model = SingleSubject(fMRI, design, clobber=True, create=True, slicetimes=[1.25]*191)

                    subject.extractors(design, model, run=run)

                    if options.fwhm or options.resels:
                        fwhmOLS = iter(FWHMestimator(subject.ref_frame, fwhm='fwhm-OLS.img', resels='resels-OLS.img', mask=subject.mask(run=run)))
                        model.fit(clobber=True, fwhmOLS=fwhmOLS, ref_contrast='filtered_func_data_overall')

                        resels, fwhm, nvoxel = subject.fwhmest.integrate(mask=subject.mask(run=run))
                        print 'Average FWHM: %0.3f' % fwhm
                        print 'Resels: %0.3f' % resels
                        print 'Number of voxels in mask: %0.3f' % nvoxel
                    else:
                        model.fit(clobber=True, ref_contrast='filtered_func_data_overall')

        if options.fixed:
            subject.getresults(resampled=True)
            for contrast in FIAC.contrasts.keys():
                for exptype in ['block', 'event']:
                    for extra in ['', '_delay']:
                        subject.within_subject('%s%s' % (contrast, extra), which=exptype)

        del(subject)
        gc.collect()

    if options.multistat:
        if not options.multiall:
            _contrast = [options.contrast]
            _which = [options.which]
            if options.delay:
                _extra = ['_delay']
            else:
                _extra = ['']
        else:
            _contrast = FIAC.contrasts.keys()
            _which = ['block', 'event']
            _extra = ['', '_delay']

        for which in _which:
            for contrast in _contrast:
                for extra in _extra:
                    print 'Attempting multistat run: "%s%s,%s"' % (contrast, extra, which)
                    try:
                        across_subjects([FIAC(subj) for subj in good_subjects], contrast='%s%s' % (contrast, extra), which=which)
                    except:
                        Warning.Warning('multistat run "%s%s,%s" failed' % (contrast, extra, which))
                        pass
        
    if options.view:
        view_results(contrast=options.contrast, stat=options.stat, which=options.which, thresh=options.thresh, anat=refs[options.ref], delay=options.delay)

    if options.view3d:
        view3d_results(contrast=options.contrast, stat=options.stat, which=options.which, thresh=options.thresh, anat=refs[options.ref], delay=options.delay)

    if options.plot:
        plot_results(contrast=options.contrast, stat=options.stat, which=options.which, thresh=options.thresh, anat=refs[options.ref], delay=options.delay, plotaxis=options.plotaxis, plotslice=options.plotslice)

    tic = time.time()
    print 'Total time for processing.py: %0.2f' % ((tic-toc)/60,)
