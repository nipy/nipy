"""
Package contains generic functions for data input/output. This includes
methods for accessing file systems and network resources.
"""

import os, gzip, bz2
from urllib2 import urlopen
from urlparse import urlparse

from neuroimaging import ensuredirs
from neuroimaging.utils.path import path

zipexts = (".gz",".bz2")
file_openers = {".gz":gzip.open, ".bz2":bz2.BZ2File, None:file}


def iszip(filename):
    filebase, ext = path(filename).splitext()
    return ext in zipexts


def splitzipext(filename):
    if iszip(filename):
        return path(filename).splitext()
    else:
        return filename, None


def unzip(filename):
    "Unzip the given file into another file.  Return the new file's name."
    if not iszip(filename): raise ValueError("file %s is not zipped"%filename)
    unzip_name, zipext = splitzipext(filename)
    opener = file_openers[zipext]
    outfile = file(unzip_name,'w')
    outfile.write(opener(filename).read())
    outfile.close()
    return unzip_name


def urlexists(url):
    try: urlopen(url)
    except: return False
    return True


def isurl(pathstr):
    scheme, netloc, _, _, _, _ = urlparse(pathstr)
    return bool(scheme and netloc)


def iswritemode(mode): return mode.find("w")>-1 or mode.find("+")>-1


class Cache (object):
            
    def __init__(self, cachepath=None):
        if cachepath is not None: 
            self.path = path(cachepath)
        elif os.name == 'posix':
            self.path = path(os.environ["HOME"]).joinpath(".nipy","cache")
        elif os.name == 'nt':
            self.path = path(os.environ["HOMEPATH"]).joinpath(".nipy","cache")
        self.setup()

    def filepath(self, uri):
        (scheme, netloc, upath, params, query, fragment) = urlparse(uri)
        return self.path.joinpath(netloc, upath[1:])

    def filename(self, uri): 
        return str(self.filepath(uri))
    
    def setup(self):
        if not self.path.exists(): ensuredirs(self.path)
        
    def cache(self, uri):
        if self.iscached(uri): return
        upath = self.filepath(uri)
        ensuredirs(upath.dirname())
        try: openedurl = urlopen(uri)
        except: raise IOError("url not found: "+str(uri))
        file(upath, 'w').write(openedurl.read())
        
    def clear(self):
        for f in self.path.files(): f.rm()
        
    def iscached(self, uri):
        return self.filepath(uri).exists()
        
    def retrieve(self, uri):
        self.cache(uri)
        return file(self.filename(uri))



class DataSource (object):

    def __init__(self, cachepath=os.curdir):
        if cachepath is not None: self._cache = Cache(cachepath)
        else: self._cache = Cache()

    def _possible_names(self, filename):
        names = (filename,)
        if not iszip(filename):
            for zipext in zipexts: names += (filename+zipext,)
        return names

    def cache(self, pathstr):
        if isurl(pathstr): self._cache.cache(pathstr)

    def filename(self, pathstr):
        found = None
        for name in self._possible_names(pathstr):
            if isurl(name) and urlexists(name):
                self.cache(name)
                found = self._cache.filename(name)
            elif path(name).exists(): found = name
            if found: break
        if found is None: raise IOError("%s not found"%pathstr)
        return found

    def exists(self, pathstr):
        try:
            _ = self.filename(pathstr)
            return True
        except IOError: return False

    def open(self, pathstr, mode='r'):
        if isurl(pathstr) and iswritemode(mode):
            raise ValueError("URLs are not writeable")
        found = self.filename(pathstr)
        _, ext = splitzipext(found)
        return file_openers[ext](found, mode=mode)



class Repository (DataSource):
    "DataSource with an implied root."
    def __init__(self, baseurl, cachepath=None):
        DataSource.__init__(self, cachepath=cachepath)
        self._baseurl = baseurl

    def _fullpath(self, pathstr):
        return path(self._baseurl).joinpath(pathstr)

    def filename(self, pathstr):
        return DataSource.filename(self, str(self._fullpath(pathstr)))

    def exists(self, pathstr):
        return DataSource.exists(self, self._fullpath(pathstr))

    def open(self, pathstr):
        return DataSource.open(self, self._fullpath(pathstr))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
