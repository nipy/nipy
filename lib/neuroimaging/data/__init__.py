import os
from path import path
#from urllib import urlopen
from urllib2 import urlopen
from urlparse import urlparse

from neuroimaging import ensuredirs

#-----------------------------------------------------------------------------
def bool(obj):
    if obj: return True
    else: return False

#-----------------------------------------------------------------------------
def urlexists(url):
    try:
        urlopen(url)
    except: return False
    return True

#-----------------------------------------------------------------------------
def isurl(pathstr):
    scheme, netloc, _,_,_,_ = urlparse(pathstr)
    return bool(scheme and netloc)


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


##############################################################################
class Repository (object):
    "Remote repository data source."
    def __init__(self, baseurl, cache=dcache):
        self._baseurl = baseurl
        self._cache = cache
    def _fullurl(pathstr):
        path(self._baseurl).joinpath(pathstr)
    def exists(self, pathstr):
        return self._cache.contains(self._fullurl(pathstr))
    def get(self, pathstr):
        return self._cache.retrieve(self._fullurl(pathstr))


##############################################################################
class FileSystem (object):
    "File system data source."
    def exists(self, pathstr): return path(pathstr).exists()
    def get(self, pathstr): return file(pathstr)


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
