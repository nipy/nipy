import os
import re
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

#-----------------------------------------------------------------------------
def iswritemode(mode):
    return mode.find("w")>-1 or mode.find("+")>-1


##############################################################################
class Cache (object):
    def __init__(self, cachepath):
        self.path = path(cachepath)
        self.setup()
    def filepath(self, uri):
        (scheme, netloc, upath, params, query, fragment) = urlparse(uri)
        return self.path.joinpath(netloc, upath[1:])
    def filename(self, uri): return str(self.filepath(uri))
    def setup(self):
        if not self.path.exists(): ensuredirs(self.path)
    def cache(self, uri):
        if self.contains(uri): return
        upath = self.filepath(uri)
        ensuredirs(upath.dirname())
        if not urlexists(uri): return
        file(upath, 'w').write(urlopen(uri).read())
    def clear(self):
        for f in self.path.files(): f.rm()
    def contains(self, uri):
        return self.filepath(uri).exists()
    def retrieve(self, uri):
        self.cache(uri)
        return file(self.filepath(uri))

# default global cache singleton
dcache = Cache(os.environ["HOME"]+"/.nipy/repository")


##############################################################################
class DataSource (object):

    def __init__(self, cache=dcache): self._cache = cache

    def filename(self, pathstr):
        if isurl(pathstr): return self._cache.filename(pathstr)
        else: return pathstr

    def exists(self, pathstr):
        if isurl(pathstr): return self._cache.contains(pathstr)
        else: return path(pathstr).exists()

    def open(self, pathstr, mode='r'):
        if isurl(pathstr):
            if iswritemode(mode): raise ValueError("URLs are not writeable")
            return self._cache.retrieve(pathstr)
        else: return file(pathstr, mode=mode)


##############################################################################
class Repository (DataSource):
    "DataSource with an implied root."
    def __init__(self, baseurl, cache=dcache):
        DataSource.__init__(self, cache=dcache)
        self._baseurl = baseurl
    def _fullpath(pathstr):
        path(self._baseurl).joinpath(pathstr)
    def filename(self, pathstr):
        return DataSource.filename(self._fullpath(pathstr))
    def exists(self, pathstr):
        return DataSource.exists(self._fullpath(pathstr))
    def open(self, pathstr):
        return DataSource.open(self, self._fullpath(pathstr))


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
