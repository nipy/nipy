import enthought.traits as traits
from spectral import spectral
import pylab

# Interpolation schemes

interpolation = traits.Trait('nearest', 'bilinear', 'blackman100',
                             'blackman256', 'blackman64', 'bicubic',
                             'sinc144', 'sinc256', 'sinc64',
                             'spline16', 'spline36')

# Color mappings available

_names = dir(pylab.cm)
_cmdict = {}
for _name in _names:
    try:
        exec('_cm = pylab.cm.%s' % _name)
        if type(_cm) == type(pylab.cm.hot):
            _cmdict[_name] = _cm
    except:
        pass

_cmdict['spectral'] = spectral

_cmapn = _cmdict.keys()
_cmapn.pop(_cmapn.index('spectral'))
_cmapn = ['spectral'] + _cmapn
cmap = apply(traits.Trait, _cmapn)

def getcmap(cmapname):
    return _cmdict[cmapname]
