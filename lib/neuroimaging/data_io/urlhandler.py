import urllib, stat, string, os, urllib2, gzip, atexit
from neuroimaging import traits
import cache


class DataFetcher(traits.HasTraits):
    """
    Not very sophisticated simple class to check if local file exists,
    otherwise, try to figure out how to get it with urllib or ftp. There is
    some checking to see if the http:// request returns a 404 or 403 error.
    """

    url = traits.Str()
    urltype = traits.Trait([None, 'file', 'http', 'ftp'] + list(string.lowercase))
    filesep = traits.Trait(['/', '\\'])
    urlhost = traits.Trait([None, traits.Str()])
    urlfile = traits.Str()
    urlbase = traits.Str()
    realm = traits.Str()
    repository = traits.Str('.')
    chmod = traits.Int(stat.S_IWUSR|stat.S_IROTH|stat.S_IRGRP|stat.S_IRUSR)
    user = traits.Str('anonymous')
    passwd = traits.Str('anonymous@')
    decompress = traits.false
    cleanup = traits.true
    clobber = traits.false
    urlstrip = traits.Str()
    otherexts = traits.ListStr()
    zipexts = traits.ListStr(['.gz'])
    cached = traits.false()


    def urldecompose(self):
        self.urltype, self.urlbase = urllib.splittype(self.url)
        self.urlhost, self.urlfile = urllib.splithost(self.urlbase)


    def urlcompose(self, type=True, urlfile=None):
        urlfile = urlfile or self.urlfile

        if self.urltype is None:
            urltype = ''
        else:
            urltype = self.urltype + ':'

        if type:
            if self.urlhost is None:
                urlhost = ''
            else:
                urlhost = '//' + self.urlhost
                return urltype + urlhost + urlfile
        else:
            if self.urlhost is not None:
                return self.urlhost + urlfile
            else:
                return urlfile


#    def __init__(self, **keywords):
#        traits.HasTraits.__init__(self, **keywords)


    def getexts(self):

        filebase, ext = os.path.splitext(self.urlfile)
        if ext in self.zipexts:
            filebase, ext = os.path.splitext(self.urlfile)

        exts = []
        for base in [ext] + self.otherexts:
            exts.append(filebase + base)

        zipexts = []
        for zipext in self.zipexts:
            for base in [ext] + self.otherexts:
                zipexts.append(filebase + base + zipext)

        exts = list(set(exts))
        zipexts = list(set(zipexts))
        return exts, zipexts
    

    def geturl(self, url, tryexts=True, force=False):

        self.url = url
        self.urldecompose()

        if os.path.exists(self.url): return self.url
        elif self.urlhost is None: raise IOError, 'file %s not found'%url

        self.cached = True
        self.urldecompose()

        outdir = os.path.join(
          self.repository,
          os.path.dirname(self.urlcompose(type=False)))
        if not os.path.exists(outdir): os.makedirs(outdir)

        if tryexts:
            exts, zipexts = self.getexts()
        else:
            exts, zipexts = [self.urlfile]
            
        goodexts = []
        for ext in exts + zipexts:
            _url = self.urlcompose(type=False, urlfile=ext)
            url = self.urlcompose(urlfile=ext)
            outname = os.path.join(self.repository, _url)

            _exists = os.path.exists(outname)
            if not _exists or force:
                check = self.urlretrieve(url, writer=file(outname, 'w'))
                if not check:
                    os.remove(outname)
                else:
                    goodexts.append(ext)
            else:
                goodexts.append(ext)

        for ext in goodexts:
            _url = self.urlcompose(type=False, urlfile=ext)
            outname = os.path.join(self.repository, _url)
            _ext = os.path.splitext(ext)[1]
            if _ext in self.zipexts:
                if _ext == '.gz':
                    if not hasattr(self, 'cache'):
                        self.cache = cache.cachedir()
                    _ungzip(outname, rm=False)

        ## TODO: have gzip uncompress into neuroimaging.cache.dirname


    def urlretrieve(self, url, writer=None):
        """
        A rudimentary check to see if urlretrieve actually finds a correct URL.

        >>> from neuroimaging.data_io.urlhandler import DataFetcher
        >>> import StringIO
        >>>
        >>> buffer = StringIO.StringIO()
        >>> fetcher = DataFetcher()
        >>> print fetcher.urlretrieve('http://kff.stanford.edu/BrainSTAT/fiac3_fonc1.txt', buffer)
        True
        >>> print buffer.len
        260
        >>>
        >>> buffer = StringIO.StringIO()
        >>> print fetcher.urlretrieve('http://www.stanford.edu/doyouexist?', buffer)
        False
        >>> print buffer.len
        0
        """
        if self.user or self.passwd:
            authhandler = urllib2.HTTPBasicAuthHandler()
            authhandler.add_password(
              self.realm, self.urlhost, self.user, self.passwd)
            opener = urllib2.build_opener(authhandler)
            urllib2.install_opener(opener)

        try:
            _url = urllib2.urlopen(url)
            _data = _url.read(10000)
            if writer is not None:
                while _data:
                    writer.write(_data)
                    _data = _url.read(100000)
                writer.close()
        except urllib2.HTTPError, e:
            if e.code == 404:
                return False
            if e.code == 403:
                return False
            raise e
    
        return True

_gzipfiles = []
_rmfiles = []


def _ungzip(outfile, clobber=True, rm=False, dir=None):
    _outfile = outfile[0:-3]

    if dir is not None:
        _outfile = os.path.join(dir, os.path.split(_outfile)[1])

    if clobber or not os.path.exists(_outfile):
        __outfile = file(_outfile, 'wb')
        gzfile = gzip.open(outfile, 'rb')
        while True:
            _data = gzfile.read(100000)
            if _data:
                __outfile.write(_data)
                __outfile.flush()
            else:
                break
            
        __outfile.close()

        if rm:
            os.remove(outfile)
            _gzipfiles.append(_outfile)
        else:
            _rmfiles.append(_outfile)


def _gzip(infile, clobber=True, rm=True):
    outfile = '%s.gz' % infile
    if clobber or not os.path.exists(outfile):
        _outfile = gzip.open(outfile, 'wb')
        infile = file(infile, 'rb')

        while True:
            _data = infile.read(100000)
            if _data:
                _outfile.write(_data)
            else:
                break

        _outfile.close()
        if rm:
            os.remove(infile)


def _gzipcleanup():
    """
    Gzip all uncompressed files and delete files whose gzipped versions were
    not deleted.
    """

    for f in _gzipfiles:
        try:
            _gzip(f, rm=True)
            print "Recompressing file: %s" % f
        except:
            pass
    for f in _rmfiles:
        try:
            os.remove(f)
            print "Removed file: %s" % f
        except:
            pass

atexit.register(_gzipcleanup)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
