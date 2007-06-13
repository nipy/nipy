import urllib2, os.path

import pylab
import numpy as N

from neuroimaging.core.api import Image
from neuroimaging.ui.visualization.viewer import BoxViewer
from scipy.sandbox.models.regression import ar_model
from neuroimaging.modalities.fmri.fmristat import delay
from neuroimaging.modalities.fmri.regression import FContrastOutput
from neuroimaging.ui.visualization import slices as vizslice
from neuroimaging.algorithms.interpolation import ImageInterpolator
from neuroimaging.ui.visualization.montage import Montage as MontageDrawer

import model, fmristat, io, fixed, multi, visualization, fiac
from protocol import eventdict_r, eventdict

class Run(model.Run):

    verbose = True
    """
    Compare results to Run
    """
    
    def __init__(self, *args, **kw):
        model.Run.__init__(self, *args, **kw)
        p = self.formula['FIAC_design']
        self.etypes = p.events.keys()
        self.indices = {}
        self.corr = {}
        self.design = fmristat.design(subj=self.subject.id, run=self.id)
        self.X = fmristat.xcache(subj=self.subject.id, run=self.id)
        self.design_match()
        self._plot_beginning()
        
    def max_corr(self):

        """
        Find and plot the signals of model.Run and compare them with
        columns of fmristat design matrix.
        """
        
        for deriv in [True, False]:
            for etype in self.etypes:
                self.indices[(deriv, etype)] = self._get_indices(deriv, etype)
                self._plot_agreement(deriv, etype)

    def fmristat_result(self, which='contrasts', contrast='speaker', stat='t'):
        """
        Retrieve fmristat result for this run
        """
        return fmristat.result(subject=self.subject.id, run=self.id,
                            which=which,
                            contrast=contrast,
                            stat=stat)

    def _get_indices(self, deriv, etype):
        """
        For a given event type (and derivative True/False), find
        corresponding columns of self.formula.design() and
        fmristat's design
        """
        
        design = self.formula['FIAC_design']
        keys = design.events.keys(); keys.sort()
        event_index = keys.index(etype)

        self.D = self.ARmodel.model().design

        if deriv:
            event_index += 4
            fmristat_index = 3
        else:
            fmristat_index = 2
    
        shift = 1.25
        T = N.arange(0, 191*2.5, 2.5) + shift

        nipy = design(T)[event_index]

        Xcol, _ = _match_cols(nipy, self.X[:,:,fmristat_index])

        Dcol, _ = _match_cols(nipy, self.D)
        cor = N.corrcoef(self.X[:,Xcol,fmristat_index], self.D[:,Dcol])[0,1]
        self.corr[(deriv, etype)] = cor, Dcol, Xcol, fmristat_index
        return Dcol, Xcol, fmristat_index

    def design_match(self):
        """
        See how close self.formula.design corresponds to
        fmristat design.
        """
        
        D = self.formula.design(N.arange(191)*2.5 + 1.25)
        for i in range(14):
            col, corr = _match_cols(self.design[:,i], D)
            if self.verbose:
                print self.formula.names()[col], corr

        for i in range(1,5):
            d = self.formula['FIAC_design'][eventdict[i]]
            d = self.hrf[0].convolve(d)
            col, corr = _match_cols(d(N.arange(191)*2.5+1.25), self.design)
            if self.verbose:
                print eventdict[i], corr, col

    def _plot_agreement(self, deriv, etype):

        design = self.formula['FIAC_design']

        keys = design.events.keys(); keys.sort()
        event_index = keys.index(etype)

        if deriv:
            event_index += 4

        def n(x):
            return x / N.fabs(x).max()

        shift = 1.25
        T = N.arange(0, 191*2.5, 2.5) + shift
        t = N.arange(0, 191*2.5, 0.02)
        
        _, _, col, index = self.corr[(deriv, etype)]
        pylab.clf()
    
        pylab.plot(T, n(self.X[:,col,index]), 'b-o',
                   label='Xcache[:,%d,%d]' % (col, index))
        pylab.plot(t, n(design(t)[event_index]), 'g', label='unshifted',
                   linewidth=2)

        a = pylab.gca()
        e = design.events[etype]

        pylab.plot(t, e(t), 'm', label='%s(=%d)' % (etype, eventdict_r[etype]),
                   linewidth=2)

        if design.design_type == 'block':
            ii = N.nonzero(e.values)[0]
            a.set_xlim([max(e.times[ii[0]]-10.,0),e.times[ii[0]+1]+40.])
            if deriv:
                a.set_ylim([-1.2,1.2])
            else:
                a.set_ylim([-0.8,1.2])

        pylab.legend()

        if design.design_type == 'event':
            a.set_xlim([0.,e.times[6]+30.])
            if deriv:
                a.set_ylim([-1.5,1.5])
            else:
                a.set_ylim([-0.8,1.6])

        pngfile = '%s/x_cache/images/compare_sub%d_run%d_deriv%d_type%s.png' % (io.data_path, self.subject.id, self.id, int(deriv), etype)
        pylab.savefig(pngfile)

    def _plot_beginning(self):

        """
        Plot 'begin' column of fmristat and model.Run -- verify visually
        that they agree.
        """
        
        b = self.formula['beginning']

        def n(x):
            return x / N.fabs(x).max()

        shift = 1.25
        T = N.arange(0, 191*2.5, 2.5) + shift
        t = N.arange(0, 191*2.5, 0.02)
        
        pylab.clf()
    
        col = 0
        index = 2
        pylab.plot(T, n(self.design[:,0]), 'b-o',
                   label='X[:,0]')
        pylab.plot(t, n(b(t)), 'g', label='unshifted',
                   linewidth=2)
        a = pylab.gca()
        a.set_ylim([-0.8,1.6])
        if self.design_type == 'block':
            a.set_xlim([0,100])
        else:
            a.set_xlim([0,20])
        pylab.legend()

        pngfile = '%s/x_cache/images/compare_sub%d_run%d_begin.png' % (io.data_path, self.subject.id, self.id)
        pylab.savefig(pngfile)

    def webpage(self):
        """
        Create a webpage with comparison plots: comparing each event type and the 'begin' event
        """
        
        self._plot_beginning()

        html = """
        <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">
        <html> <head>
        <title></title>
        </head>

        <body>
        """

        html += """
        <h1>Comparing X_cache for subject %d, run %d</h1>\n
        """ % (self.subject.id, self.id)
            
        for etype in self.etypes:
            for deriv in [False, True]:
                pngfile = 'compare_sub%d_run%d_deriv%d_type%s.png' % (self.subject.id, self.id, int(deriv), etype)
                html += """
                <h2>Event type %s, derivative=%s</h2>\n'  
                <img src="images/%s">
                """ % (etype, str(deriv), pngfile)

        html += """
        <h2>Beginning</h2>
        <img src="images/%s>
        <hr>
        <address></address>
        <!-- hhmts start --> <!-- hhmts end -->
        </body> </html>
        """ % 'compare_sub%d_run%d_begin.png' % (self.subject.id, self.id)

        htmlfile = file('%s/x_cache/compare_sub%d_run%d.html' % (io.data_path, self.subject.id, self.id), 'w')
        htmlfile.write(html)
        htmlfile.close()

class Voxel(Run):

    """
    Compare results at a particular voxel using voxel_model methodd.
    """
    
    def voxel_model(self, i, j, k):
        self.load()
        rho = fmristat.rho(subject=self.subject.id, run=self.id)[i,j,k]        
        self.rho = N.around(rho * (self.OLSmodel.nmax / 2.)) / (self.OLSmodel.nmax / 2.)
        parcelmap, parcelseq = self.OLSmodel.getparcelmap()

        mask = N.equal(parcelmap, self.rho)

        self.AR(clobber=True, parcel=(parcelmap, [self.rho]))
        self.design = self.OLSmodel.model().design

        m = ar_model(self.design, self.rho)
        self.load()
        self.fmri.postread = self.brainavg
        data = self.fmri[:,i,j,k]
        results = m.fit(data)

        cor = {}

        for output in self.ARmodel.outputs:
            if isinstance(output, delay.DelayContrastOutput):
                RR = {}
                KK = {}
                for stat in ['effect', 'sd', 't']:
                    RR[stat] = []
                    KK[stat] = []
                for _j in range(4):
                    for stat in ['effect', 'sd', 't']:
                        R = self.result(which='delays',
                                        contrast=output.contrast.rownames[_j],
                                        stat=stat)
                        K = self.fmristat_result(which='delays',
                                              contrast=output.contrast.rownames[_j],
                                              stat=stat)
                        cor[('delays', stat, output.contrast.rownames[_j])] = N.corrcoef(R[:].flat * mask[:], K[:].flat * mask[:])[0,1]
                        RR[stat].append(R[i,j,k])
                        KK[stat].append(K[i,j,k])

                for stat in ['effect', 'sd', 't']:
                    for obj in [RR,KK]:
                        obj[stat] = N.array(obj[stat])
                if self.verbose:
                    print output.extract(results), RR, KK

            elif not isinstance(output, FContrastOutput):
                RR = {}
                KK = {}
                for stat in ['effect', 'sd', 't']:
                    R = self.result(which='contrasts', contrast=output.contrast.name, stat=stat)
                    K = self.fmristat_result(which='contrasts', contrast=output.contrast.name, stat=stat)
                    RR[stat] = R[i,j,k]
                    KK[stat] = K[i,j,k]

                    cor[('contrasts', stat, output.contrast.name)] = N.corrcoef(R[:].flat * mask[:], K[:].flat * mask[:])[0,1]

                if self.verbose:
                    print output.extract(results), RR, KK, output.contrast.name
            else:
                if self.verbose:
                    print output.extract(results)
        if self.verbose:
            print cor
        
class FmristatRun(Run):

    """
    Run the analysis replacing estimate of AR(1) parameter
    with fmristat's estimate for second pass of model.
    """

    def OLS(self, **OLSopts):
        CompareRun.OLS(self, **OLSopts)
        self.OLSmodel.rho = fmristat.rho(subject=self.subject.id, run=self.id)

class FixedSubject(fixed.Subject):

    def result(self, stat='t', resampled=True):
        return fmristat.fixed(subject=self.id,
                              which=self.study.which,
                              contrast=self.study.contrast,
                              stat=stat,
                              design=self.study.design)
    def estimates(self):
        v = [self.result(stat=stat) for stat in ['effect', 'sd', 't']]
        return {'effect': v[0], 'sd': v[1], 't': v[2]}


class VisualizationFixed(visualization.Fixed):

    def _get_images(self):
        vmin = []; vmax = []
        self.images = {}
        for s in fiac.subjects:
            im = Image(fmristat.fixed(subject=s,
                                      which=self.which,
                                      contrast=self.contrast,
                                      stat=self.stat,
                                      design=self.design))
            fixed.set_transform(im)
            self.images[s] = visualization.Slice(im, axis='fixed')
            d = self.images[s].dslice.scalardata()
            self.images[s].dslice.transpose = True
            vmin.append(N.nanmin(d)); vmax.append(N.nanmax(d))

        self.vmin = N.array(vmin).mean(); self.vmax = N.array(vmax).mean()
        
    def output(self):
        pylab.savefig(self.resultpath('%s_fmristat.png' % self.stat))

def visualization_run(contrast='average', which='contrasts', design='event'):

    for stat in ['effect', 'sd', 't']:
        v = VisualizationFixed(root=io.data_path,
                  stat=stat,
                  which=which,
                  contrast=contrast,
                  design=design)
        if stat == 't':
            v.vmax = 4.5; v.vmin = -4.5
        v.draw()
        v.output()
        
        htmlfile = file(v.resultpath("fmristat.html"), 'w')
        htmlfile.write("""
        <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">
        <html> <head>
        <title></title>
        </head>
        
        <body>
        <h2>Contrast %s, %s design, %s</h2>
        <h3>Effect</h3>
        <img src="effect.png">
        <h3>Effect, fmristat</h3>
        <img src="effect_fmristat.png">
        <h3>SD</h3>
        <img src="sd.png">
        <h3>SD, fmristat</h3>
        <img src="sd_fmristat.png">
        <h3>T</h3>
        <img src="t.png">
        <h3>T, fmristat</h3>
        <img src="t_fmristat.png">
        </body>
        </html>
        """ % (contrast, design, {'contrasts': 'magnitude', 'delays':'delay'}[which]))
        htmlfile.close()
        del(v)

def _match_cols(vector, matrix):
    
    """
    For given (vector, matrix), find maximal correlation (and its index)
    of vector with columns of matrix.
    """
    
    n = matrix.shape[1]
    v = N.zeros((n,))
    for i in range(n):
        v[i] = N.corrcoef(vector, matrix[:,i])[0,1]
    return N.argmax(N.fabs(N.nan_to_num(v))), N.max(N.fabs(N.nan_to_num(v)))

def _cor(im1, im2, mask=1.):
    """
    Return correlation of two images.
    """
    x = N.nan_to_num(im1[:])
    y = N.nan_to_num(im2[:])
    m = mask[:]
    x *= m
    y *= m
    x.shape = N.product(x.shape)
    y.shape = N.product(y.shape)
    return N.corrcoef(x, y)[0,1]

