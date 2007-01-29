import numpy as N
import scipy.io
import scipy.optimize
import pylab, os
import urllib2

import run as FIACrun

shift = 1.25
T = N.arange(0, 191*2.5, 2.5) + shift
t = N.arange(0, 191*2.5, 0.02)

def n(x): return x / N.fabs(x).max()

def getxcache(subj=0, run=1):
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
    X = scipy.io.loadmat(x_cache)['X']
    if len(X.shape) == 4:
        X = X[:,:,:,0]
    return X

def getfmristat(subj=0, run=1):
    X = getxcache(subj=subj, run=run)[:,:,2:] * 1.
    X.shape = (191, 10)
    return X
    
def getnipy(subj=0, run=1):
    f = FIACrun.FIACformula(subj=subj, run=run)
    d = f['FIAC_design'] + f['beginning']
    return d(time=T)

def compare(subj=0, run=1, show=False, save=True, etype='DSt_DSp', deriv=True):

    f = FIACrun.FIACformula(subj=subj, run=run)

    """
    First, get the formula term corresponding to the design,
    and find out
    """
    
    design = f['FIAC_design']
    event_index = design._event_keys.index(etype)

    X = getxcache(subj=subj, run=run)

    if deriv:
        event_index += 4
        fmristat_index = 3  
    else:
        fmristat_index = 2  
        
    """
    Find the correspondence between the columns.
    """
    
    v = N.zeros((4,), N.float64)
    for i in range(4):
        v[i] = ((n(design(T)[event_index]) - n(X[:,i,fmristat_index]))**2).sum()

    fmristat_col = N.argmin(v)

    pylab.clf()
    
    pylab.plot(T, n(X[:,fmristat_col,fmristat_index]), 'b-o', label='Xcache[:,%d,%d]' % (fmristat_col, fmristat_index))
    pylab.plot(t, n(design(t)[event_index]), 'g', label='unshifted', linewidth=2)

    a = pylab.gca()
    e = design.events[etype]

    pylab.plot(t, e(t), 'm', label='%s(=%d)' % (etype, FIACrun.eventdict_r[etype]), linewidth=2)

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

    pngfile = '/home/analysis/FIAC/x_cache/images/compare_sub%d_run%d_deriv%d_type%s.png' % (subj, run, int(deriv), etype)
    if save:
        pylab.savefig(pngfile)
    if show:
        pylab.show()
    return pngfile

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

    for i in [5] + range(16):
        for j in range(1, 5):
            htmlfile = file('/home/analysis/FIAC/x_cache/compare_sub%d_run%d.html' %(i,j), 'w')
            htmlfile.write(html1)
            htmlfile.write('<h1>Comparing X_cache for subject %d, run %d</h1>\n' % (i, j))
            for etype in etypes:
                for deriv in [False, True]:
                    hasany = True
                    try:
                        pngfile = compare(subj=i, run=j, deriv=deriv, etype=etype)
                        htmlfile.write('<h2>Event type %s, derivative=%s</h2>\n' % (etype, str(deriv)))
                        htmlfile.write('<img src="images/%s">' % os.path.basename(pngfile))
                        print i, j, deriv, etype
                    except urllib2.HTTPError:
                        hasany = False
                        pass
                    except NotImplementedError:
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
