import BrainSTAT, os

keithdir = 'http://www.math.mcgill.ca/keith/jonathan/'
jonathandir = 'http://kff.stanford.edu/FIAC/'

mapping = {'overall': ('overall', 'all'),
           'sentence': ('sentence', 'sen'),
           'speaker': ('speaker', 'spk'),
           'interaction': ('interaction', 'snp')}

def keithfixed(subject, contrast='sentence', delay=False, stat='t', which='block'):
    if which is 'block':
        which = 'bloc'
    else:
        raise ValueError, "Keith hasn't done the events yet!"

    if stat == 'eff':
        stat = 'ef'

    if delay:
        delay = 'del'
    else:
        delay = 'mag'

    return BrainSTAT.VImage(os.path.join(keithdir, 'subj%d_%s_%s_%s_%s.img' % (subject, which, mapping[contrast][1], delay, stat)),urlstrip='/keith/jonathan/')

def jonathanfixed(subject, contrast='sentence', delay=False, stat='t', which='block'):
    if delay:
        delay = '_delay'
    else:
        delay = ''

    return BrainSTAT.VImage(os.path.join(jonathandir, 'fiac%d/fixed' % subject, '%s%s_%s_%s.img' % (mapping[contrast][0], delay, which, stat)), realm='FIAC Website', username='keith', password='poincare', urlstrip='/FIAC/fiac%d/fixed/' % subject)

contrast = 'sentence'
delay = True
subject = 3
stat = 'sd'
vmin = 0
vmax = 9.
import pylab

jonathanfixed(subject, contrast=contrast, delay=delay, stat=stat).view(show=False, vmin=vmin, vmax=vmax)
pylab.title('jonathan')
pylab.figure()
keithfixed(subject, contrast=contrast, delay=delay, stat=stat).view(show=False, vmin=vmin, vmax=vmax)
pylab.title('keith')

pylab.show()
