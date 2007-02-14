import urllib2, os.path

import pylab
import numpy as N

from neuroimaging.ui.visualization.viewer import BoxViewer
from scipy.sandbox.models.regression import ar_model
from neuroimaging.modalities.fmri.fmristat import delay
from neuroimaging.modalities.fmri.regression import FContrastOutput
from neuroimaging.ui.visualization import slices as vizslice
from neuroimaging.algorithms.interpolation import ImageInterpolator
from neuroimaging.ui.visualization.montage import Montage as MontageDrawer

import model, fmristat, io
from protocol import eventdict_r, eventdict

class CompareRun(model.RunModel):

    def __init__(self, *args, **kw):
        model.RunModel.__init__(self, *args, **kw)
        p = self.formula['FIAC_design']
        self.etypes = p.events.keys()
        self.indices = {}
        self.corr = {}
        self.design = fmristat.design(subj=self.subject.id, run=self.id)
        self.X = fmristat.xcache(subj=self.subject.id, run=self.id)
        self.design_match()
        self._plot_beginning()
        
    def max_corr(self):

        for deriv in [True, False]:
            for etype in self.etypes:
                self.indices[(deriv, etype)] = self._get_indices(deriv, etype)
                self._plot_agreement(deriv, etype)

    def fmristat_result(self, which='contrasts', contrast='speaker', stat='t'):
        return fmristat.result(subject=self.subject.id, run=self.id,
                            which=which,
                            contrast=contrast,
                            stat=stat)

    def _get_indices(self, deriv, etype):
        """
        Verify that space spanned by design columns is the same as fMRIstat --
        or that the columns are basically the same.
        """
        
        """
        First, get the formula term corresponding to the design,
        and find out
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
        """
        Find the correspondence between the columns.
        """
    
        shift = 1.25
        T = N.arange(0, 191*2.5, 2.5) + shift

        nipy = design(T)[event_index]

        Xcol, _ = self._match_cols(nipy, self.X[:,:,fmristat_index])

        Dcol, _ = self._match_cols(nipy, self.D)
        cor = N.corrcoef(self.X[:,Xcol,fmristat_index], self.D[:,Dcol])[0,1]
        self.corr[(deriv, etype)] = cor, Dcol, Xcol, fmristat_index
        return Dcol, Xcol, fmristat_index

    def design_match(self):
        D = self.formula.design(N.arange(191)*2.5 + 1.25)
        for i in range(14):
            col, corr = self._match_cols(self.design[:,i], D)
            print self.formula.names()[col], corr

        for i in range(1,5):
            d = self.formula['FIAC_design'][eventdict[i]]
            d = self.hrf[0].convolve(d)
            col, corr = self._match_cols(d(N.arange(191)*2.5+1.25), self.design)
            print eventdict[i], corr, col

    def _match_cols(self, vector, matrix):

        n = matrix.shape[1]
        v = N.zeros((n,))
        for i in range(n):
            v[i] = N.corrcoef(vector, matrix[:,i])[0,1]
        return N.argmax(N.fabs(N.nan_to_num(v))), N.max(N.fabs(N.nan_to_num(v)))

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
        if self.type == 'block':
            a.set_xlim([0,100])
        else:
            a.set_xlim([0,20])
        pylab.legend()

        pngfile = '%s/x_cache/images/compare_sub%d_run%d_begin.png' % (io.data_path, self.subject.id, self.id)
        pylab.savefig(pngfile)

    def webpage(self):

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
            
        for etype in runmodel.etypes:
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

class VoxelModel(CompareRun):

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

                print output.extract(results), RR, KK, output.contrast.name
            else:
                print output.extract(results)
        print cor
        
class FmristatRho(CompareRun):

    def OLS(self, **OLSopts):
        CompareRun.OLS(self, **OLSopts)
        self.OLSmodel.rho = fmristat.rho(subject=self.subject.id, run=self.id)


def plot_result(result1, result2, mask, which='contrasts', contrast='average', stat='t', vmax=None, vmin=None):
    
    resultargs = {'which':which, 'contrast':contrast, 'stat':stat}
    x = result1(**resultargs)
    y = result2(**resultargs)

    vx = BoxViewer(x, mask=mask, colormap='spectral')
    vy = BoxViewer(y, mask=mask, colormap='spectral')

    if vmin is not None:
        vx.m = vmin
        vy.m = vmin

    if vmax is not None:
        vx.M = vmax
        vy.M = vmax

    vx.draw()
    vy.draw()

    X = x.readall() * mask.readall()
    X.shape = N.product(X.shape)

    Y = y.readall() * mask.readall()
    Y.shape = N.product(Y.shape)

    print 'corr', N.corrcoef(X, Y)[0,1]
    pylab.show()

def cor_result(result1, result2, mask, which='contrasts', contrast='speaker', stat='t'):
    resultargs = {'which':which, 'contrast':contrast, 'stat':stat}
    x = result1(**resultargs)
    y = result2(**resultargs)

    x = N.nan_to_num(x[:]) * mask[:]
    y = N.nan_to_num(y[:]) * mask[:]
    x.shape = N.product(x.shape)
    y.shape = N.product(y.shape)
    return N.corrcoef(x, y)


