import urllib2, os.path

import pylab
import numpy as N

import model, keith
from protocol import eventdict_r
from neuroimaging.core.image.image import Image
from neuroimaging.ui.visualization.viewer import BoxViewer
from scipy.sandbox.models.regression import ols_model, ar_model
from neuroimaging.modalities.fmri.fmristat import delay
from neuroimaging.modalities.fmri.regression import FContrastOutput

class CompareRun(model.RunModel):

    def __init__(self, *args, **kw):
        model.RunModel.__init__(self, *args, **kw)
        p = self.formula['FIAC_design']
        self.etypes = p.events.keys()
        self.indices = {}
        self.corr = {}

    def max_corr(self):

        for deriv in [True, False]:
            for etype in self.etypes:
                self.indices[(deriv, etype)] = self._get_indices(deriv, etype)
                self._plot_agreement(deriv, etype)

    def result(self, which='contrasts', contrast='speaker', stat='t'):
        resultfile = os.path.join(self.resultdir, which, contrast, "%s.img" % stat)
        return Image(resultfile)

    def keith_result(self, which='contrasts', contrast='speaker', stat='t'):
        return keith.result(subject=self.subject.id, run=self.id,
                            which=which,
                            contrast=contrast,
                            stat=stat)

    def _get_indices(self, deriv, etype):
        """
        Verify that space spanned by design columns is the same as fMRIstat --
        or that the columns are basically the same.
        """
        
        def n(x): return x / N.fabs(x).max()

        """
        First, get the formula term corresponding to the design,
        and find out
        """
    
        design = self.formula['FIAC_design']
        event_index = design._event_keys.index(etype)

        self.X = keith._getxcache(subj=self.subject.id, run=self.id)
        self.D = self.ARmodel.model().design

        if deriv:
            event_index += 4
            index = 3  
        else:
            index = 2  
        
        """
        Find the correspondence between the columns.
        """
    
        shift = 1.25
        T = N.arange(0, 191*2.5, 2.5) + shift
        v = N.zeros((4,))

        nipy = design(T)[event_index]

        if deriv:
            fmristat_index = 3
        else:
            fmristat_index = 2
        Xcol = self._match_cols(nipy, self.X[:,:,fmristat_index])

        Dcol = self._match_cols(nipy, self.D)
        cor = N.corrcoef(self.X[:,Xcol,fmristat_index], self.D[:,Dcol])[0,1]
        self.corr[(deriv, etype)] = cor, Dcol, Xcol, fmristat_index
        return Dcol, Xcol, fmristat_index

    def _match_cols(self, vector, matrix):

        n = matrix.shape[1]
        v = N.zeros((n,))
        for i in range(n):
            v[i] = N.corrcoef(vector, matrix[:,i])[0,1]
        return N.argmax(N.fabs(N.nan_to_num(v)))

    def _plot_agreement(self, deriv, etype):

        design = self.formula['FIAC_design']
        event_index = design._event_keys.index(etype)
        if deriv:
            event_index += 4

        def n(x): return x / N.fabs(x).max()

        shift = 1.25
        T = N.arange(0, 191*2.5, 2.5) + shift
        t = N.arange(0, 191*2.5, 0.02)
        
        _, _, col, index = self.corr[(deriv, etype)]
        pylab.clf()
    
        pylab.plot(T, n(self.X[:,col,index]), 'b-o', label='Xcache[:,%d,%d]' % (col, index))
        pylab.plot(t, n(design(t)[event_index]), 'g', label='unshifted', linewidth=2)

        a = pylab.gca()
        e = design.events[etype]

        pylab.plot(t, e(t), 'm', label='%s(=%d)' % (etype, eventdict_r[etype]), linewidth=2)

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

        pngfile = '/home/analysis/FIAC/x_cache/images/compare_sub%d_run%d_deriv%d_type%s.png' % (self.subject.id, self.id, int(deriv), etype)
        pylab.savefig(pngfile)

    def webpage(self):
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
        <hr>
        <address></address>
        <!-- hhmts start --> <!-- hhmts end -->
        </body> </html>
        """
        htmlfile = file('/home/analysis/FIAC/x_cache/compare_sub%d_run%d.html' % (self.subject.id, self.id), 'w')
        htmlfile.write(html)
        htmlfile.close()

class VoxelModel(CompareRun):

    def voxel_model(self, i, j, k):
        self.load()
        rho = keith.rho(subject=self.subject.id, run=self.id)[i,j,k]        
        self.rho = N.around(rho * (self.OLSmodel.nmax / 2.)) / (self.OLSmodel.nmax / 2.)
        parcelmap, parcelseq = self.OLSmodel.getparcelmap()

        mask = N.equal(parcelmap, self.rho)

        self.AR(clobber=True, parcel=(parcelmap, [self.rho]))
        self.design = self.OLSmodel.model().design

        self.X = keith._getxcache(subj=self.subject.id, run=self.id)

        m = ar_model(self.design, self.rho)
        self.load()
        self.fmri.postread = self.brainavg
        data = self.fmri[:,i,j,k]
        results = m.fit(data)

        cor = {}

        for output in self.ARmodel.outputs:
            if isinstance(output, delay.DelayContrastOutput):
                RR = {}; KK = {}
                for stat in ['effect', 'sd', 't']:
                    RR[stat] = []; KK[stat] = []
                for _j in range(4):
                    for stat in ['effect', 'sd', 't']:
                        R = self.result(which='delays', contrast=output.contrast.rownames[_j], stat=stat)
                        K = self.keith_result(which='delays', contrast=output.contrast.rownames[_j], stat=stat)
                        cor[('delays', stat, output.contrast.rownames[_j])] = N.corrcoef(R[:].flat * mask[:], K[:].flat * mask[:])[0,1]
                        RR[stat].append(R[i,j,k]); KK[stat].append(K[i,j,k])

                for stat in ['effect', 'sd', 't']:
                    for obj in [RR,KK]:
                        obj[stat] = N.array(obj[stat])
                print output.extract(results), RR, KK

            elif not isinstance(output, FContrastOutput):
                RR = {}
                KK = {}
                for stat in ['effect', 'sd', 't']:
                    R = self.result(which='contrasts', contrast=output.contrast.name, stat=stat)
                    K = self.keith_result(which='contrasts', contrast=output.contrast.name, stat=stat)
                    RR[stat] = R[i,j,k]
                    KK[stat] = K[i,j,k]

                    cor[('contrasts', stat, output.contrast.name)] = N.corrcoef(R[:].flat * mask[:], K[:].flat * mask[:])[0,1]

                print output.extract(results), RR, KK, output.contrast.name
            else:
                print output.extract(results)
        print cor
        
class KeithRho(CompareRun):

    def OLS(self, **OLSopts):
        CompareRun.OLS(self, **OLSopts)
        self.OLSmodel.rho = keith.rho(subject=self.subject.id, run=self.id)

def plot_result(result1, result2, mask, which='contrasts', contrast='speaker', stat='t', vmax=None, vmin=None):
    
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

    x = N.nan_to_num(x[:]) * mask[:]; x.shape = N.product(x.shape)
    y = N.nan_to_num(y[:]) * mask[:]; y.shape = N.product(y.shape)
    return N.corrcoef(x, y)


if __name__ == '__main__':

    import sys
    if len(sys.argv) == 3:
        subj, run = map(int, sys.argv[1:])
    else:
        subj, run = (3, 1)

    study = model.StudyModel(root='/home/analysis/FIAC')
    subject = model.SubjectModel(subj, study=study)

##     try:
##         runmodel.max_corr()
##         runmodel.webpage()
##     except (urllib2.HTTPError, ValueError, NotImplementedError):
##         pass

##     print runmodel.corr
##     runmodel.load()
##     mask = runmodel.mask


##     vmodel = VoxelModel(subject, run)
##     vmodel.OLS(clobber=True)
##     vmodel.voxel_model(15,32,20)


    runmodel = CompareRun(subject, run)
    runmodel.OLS(clobber=True)
    runmodel.AR(clobber=True)
        

##     keithrho = KeithRho(subject, run, resultdir=os.path.join("fsl", "fmristat_rho"))
##     keithrho.OLS(clobber=True)
##     keithrho.AR(clobber=True)

    M = runmodel
    M.load()
    for contrast in ['average', 'sentence', 'speaker', 'interaction']:
        v = [cor_result(M.result, M.keith_result, M.mask, stat=stat,
                        contrast=contrast)[0,1] for stat in ['t', 'sd', 'effect']]
        r = (N.nan_to_num(M.result(contrast=contrast, stat='sd')[:]) / 
             N.nan_to_num(M.keith_result(contrast=contrast, stat='sd')[:])) * M.mask[:]
        r = r[N.nonzero(N.nan_to_num(r))]
        print v, r.mean(), r.std(), r.min(), r.max()



##     plot_result(runmodel.result, runmodel.keith_result, runmodel.mask)
##     rho1 = runmodel.OLSmodel.rho
##     rho2 = keithrho.OLSmodel.rho


