import os
from path import path
from urllib import urlopen
from urlparse import urlparse


##############################################################################
class Cache (object):
    def __init__(self, cachepath):
        self.path = path(cachepath)
    def _uripath(self, uri):
        (scheme, netloc, upath, params, query, fragment) = urlparse(uri)
        return self.path/netloc/upath
    def setup(self):
        if not self.path.exists(): self.path.makedirs()
    def clear(self):
        for f in self.path.files(): f.rm()
    def contains(self, uri):
        return self._uripath(uri).exists()
    def retrieve(self, uri):
        return file(self._uripath(uri))

# default global cache singleton
cache = Cache(os.environ["HOME"]+"/.nipy/repository")


##############################################################################
class Repository (object):
    def __init__(self, cache=cache): self.cache = cache
    def retrieve(self, uri):
        if self.cache.contains(uri): return self.cache.retrieve(uri)
        else: return urlopen(uri)

#-----------------------------------------------------------------------------
def retrieve(uri):
    """
    >>> f = retrieve('http://kff.stanford.edu/~jtaylo/BrainSTAT/rho.img')
    >>> len(f.read())
    851968
    """
    return Repository().retrieve(uri)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
