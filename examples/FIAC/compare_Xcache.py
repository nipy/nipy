
import numpy as N
import scipy.io
import scipy.optimize
import pylab, os
import urllib2

import run as FIACrun

shift = 1.25
T = N.arange(0, 191*2.5,2.5) + shift
t = N.arange(0,191*2.5,0.02)

def n(x): return x / N.fabs(x).max()

def getxcache(subj=0, run=1):
    mat = urllib2.urlopen('http://kff.stanford.edu/FIAC/x_cache/mat/x_cache_sub%d_run%d.mat' % (subj, run)).read()
    if not os.path.exists('x_cache'):
        os.makedirs('x_cache')
    outfile = file('x_cache/x_cache_sub%d_run%d.mat' % (subj, run), 'wb')
    outfile.write(mat)
    outfile.close()
    

def compare(subj=0, run=1, show=False, save=True, etype='DSt_DSp', deriv=True):

    f = FIACrun.FIACformula(subj=subj,run=run)

    pi = f.termnames().index('FIAC_design')
    p = f.terms[pi]

    x_cache = 'x_cache/x_cache_sub%d_run%d.mat' % (subj, run)
    if not os.path.exists(x_cache):
        getxcache(subj=subj, run=run)
    X = scipy.io.loadmat(x_cache)['X']
    if len(X.shape) == 4:
        X = X[:,:,:,0]

    pi = p._event_keys.index(etype)
    if deriv:
        dpi = pi + 4
        kfi = 3
    else:
        dpi = pi
        kfi = 2
        
    v = N.zeros((4,), N.Float)

    for i in range(4):
        v[i] = ((n(p(T)[dpi]) - n(X[:,i,kfi]))**2).sum()

    ki = N.argmin(v)

    pylab.clf()
    
    pylab.plot(T, n(X[:,ki,kfi]), 'b-o', label='Xcache[:,%d,%d]' % (ki, kfi))
    pylab.plot(T, n(X[:,ki,kfi-2]), 'r-o', label='Xcache[:,%d,%d]' % (ki, kfi-2))
    pylab.plot(t, n(p(t+shift)[dpi]), 'g', label='unshifted', linewidth=2)
    
    def crit(x, dx=1.0e-05):
        v = ((n(p(T+x)[dpi]) - n(X[:,ki,kfi]))**2).sum()
        w = ((n(p(T+x+dx)[dpi]) - n(X[:,ki,kfi]))**2).sum()
        return v-w

    dopt = scipy.optimize.bisect(crit, -1.5, 1.5)

    pylab.plot(t, n(p(t+dopt)[dpi]), 'g--', label='shifted', linewidth=2)

    a = pylab.gca()
    e = p.events[etype]
    print e.name

    pylab.plot(t, e(t), 'm', label='%s(=%d)' % (etype, FIACrun.eventdict_r[etype]), linewidth=2)

    hrf = FIACrun.DelayHRF()
    ee = hrf.convolve(e, interval=[-5,192*2.5])[int(deriv)]
    
    if p.design_type == 'block':
        ii = N.nonzero(e.values)
        a.set_xlim([max(e.times[ii[0]]-10.,0),e.times[ii[0]+1]+40.])
        if deriv:
            a.set_ylim([-1.2,1.2])
        else:
            a.set_ylim([-0.8,1.2])

    pylab.legend()

    if p.design_type == 'event':
        a.set_xlim([0.,e.times[6]+30.])
        if deriv:
            a.set_ylim([-1.5,1.5])
        else:
            a.set_ylim([-0.8,1.6])

    pylab.title('Best shift: %0.3f' % dopt)
    pngfile = '/home/analysis/FIAC/x_cache/images/compare_sub%d_run%d_deriv%d_type%s.png' % (subj, run, int(deriv), etype)
    if save:
        pylab.savefig(pngfile)
    if show:
        pylab.show()
    return dopt, pngfile

shifts = []

f = FIACrun.FIACformula()
pi = f.termnames().index('FIAC_design')
p = f.terms[pi]
etypes = p.events.keys()

html1 = """
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML//EN">
<html> <head>
<title></title>
</head>

<body>
"""

html2 = """
<hr>
<address></address>
<!-- hhmts start --> <!-- hhmts end -->
</body> </html>
"""

if __name__ == '__main__':
    for i in range(16):
        for j in range(1,5):
            htmlfile = file('/home/analysis/FIAC/x_cache/compare_sub%d_run%d.html' %(i,j), 'w')
            htmlfile.write(html1)
            htmlfile.write('<h1>Comparing X_cache for subject %d, run %d</h1>\n' % (i, j))
            for etype in etypes:
                for deriv in [False, True]:
                    try:
                        hasany = True
                        dopt, pngfile = compare(subj=i, run=j, deriv=deriv, etype=etype)
                        shifts.append(dopt)
                        htmlfile.write('<h2>Event type %s, derivative=%s</h2>\n' % (etype, str(deriv)))
                        htmlfile.write('<img src="images/%s">' % os.path.basename(pngfile))
                        print i, j, deriv, etype
                    except:
                        hasany = False
                        pass
            htmlfile.write(html2)

            htmlfile.close()
            if not hasany:
                os.remove(htmlfile.name)

    sfile = file('shifts.csv', 'w')
    import csv
    writer = csv.writer(sfile)
    for shift in shifts:
        writer.writerow([shift])
    sfile.close()
