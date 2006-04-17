import os
from path import path
#from urllib import urlopen
from urllib2 import urlopen
from urlparse import urlparse

from neuroimaging import ensuredirs

#-----------------------------------------------------------------------------
def urlexists(url):
    try:
        urlopen(url)
    except: return False
    return True


##############################################################################
class Cache (object):
    def __init__(self, cachepath):
        self.path = path(cachepath)
        self.setup()
    def uripath(self, uri):
        (scheme, netloc, upath, params, query, fragment) = urlparse(uri)
        return self.path.joinpath(netloc, upath[1:])
    def setup(self):
        if not self.path.exists(): ensuredirs(self.path)
    def cache(self, uri):
        if self.contains(uri): return
        upath = self.uripath(uri)
        ensuredirs(upath.dirname())
        if not urlexists(uri): return
        file(upath, 'w').write(urlopen(uri).read())
    def clear(self):
        for f in self.path.files(): f.rm()
    def contains(self, uri):
        return self.uripath(uri).exists()
    def retrieve(self, uri):
        self.cache(uri)
        return file(self.uripath(uri))

# default global cache singleton
dcache = Cache(os.environ["HOME"]+"/.nipy/repository")


#-----------------------------------------------------------------------------
def retrieve(uri, cache=dcache):
    """
    >>> f = retrieve('http://kff.stanford.edu/~jtaylo/BrainSTAT/rho.img')
    >>> len(f.read())
    851968
    """
    return cache.retrieve(uri)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
