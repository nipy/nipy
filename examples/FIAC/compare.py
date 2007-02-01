import urllib2

import pylab
import numpy as N

import model, keith
from protocol import eventdict_r

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
        v = N.zeros((4,), N.float64)

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
        v = N.zeros((n,), N.float64)
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
        print htmlfile


if __name__ == '__main__':

    import sys
    if len(sys.argv) == 3:
        subj, run = map(int, sys.argv[1:])
    else:
        subj, run = (3, 1)

    study = model.StudyModel(root='/home/analysis/FIAC')
    subject = model.SubjectModel(subj, study=study)
    runmodel = CompareRun(subject, run)
    runmodel.OLS(clobber=True)
    runmodel.AR(clobber=True)

    try:
        runmodel.max_corr()
        runmodel.webpage()
        print runmodel.corr
    except (urllib2.HTTPError, ValueError, NotImplementedError):
        pass

