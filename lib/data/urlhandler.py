import urllib, stat, string, sets, os, urllib2, gzip
import enthought.traits as traits

class DataFetcher(traits.HasTraits):
    """
    Not very sophisticated simple class to check if local file exists, otherwise, try to figure out how to get it with urllib or ftp. There is some checking to see if the http:// request returns a 404 or 403 error.
    
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

        if urlfile is None:
            urlfile = self.urlfile

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

    def __init__(self, **keywords):
        traits.HasTraits.__init__(self, **keywords)

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

        exts = list(sets.Set(exts))
        zipexts = list(sets.Set(zipexts))
        
        return exts, zipexts
    
    def geturl(self, url, tryexts=True, force=False):

        self.url = url
        self.urldecompose()

        if os.path.exists(self.url):
            return self.url

        self.cached = True
        self.urldecompose()

        outdir = os.path.join(self.repository, os.path.dirname(self.urlcompose(type=False)))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if tryexts:
            exts, zipexts = self.getexts()
        else:
            exts, zipexts = [self.urlfile]
            
        rmfiles = []
        goodexts = []
        for ext in exts + zipexts:
            _url = self.urlcompose(type=False, urlfile=ext)
            url = self.urlcompose(urlfile=ext)
            outname = os.path.join(self.repository, _url)
            if ext in zipexts:
                _outname, junk = os.path.splitext(outname)
                _exists = os.path.exists(_outname)
            else:
                _exists = os.path.exists(outname)
            if not _exists or force:
                check = self.urlretrieve(url, writer=file(outname, 'w'))
                if not check:
                    os.remove(outname)
                else:
                    goodexts.append(ext)

        for ext in goodexts:
            _url = self.urlcompose(type=False, urlfile=ext)
            outname = os.path.join(self.repository, _url)
            _ext = os.path.splitext(ext)[1]
            if _ext in self.zipexts:
                if _ext == '.gz':
                    ungzip(outname, rm=True)

    def urlretrieve(self, url, writer=None):

        """

        A rudimentary check to see if urlretrieve actually finds a correct URL.

        >>> from BrainSTAT.Base.Pipes import _urlretrieve
        >>> import StringIO
        >>>
        >>> buffer = StringIO.StringIO()
        >>> print _urlretrieve('http://kff.stanford.edu/BrainSTAT/fiac3_fonc1.txt', buffer)
        True
        >>> print buffer.len
        260
        >>>
        >>> buffer = StringIO.StringIO()
        >>> print _urlretrieve('http://www.stanford.edu/doyouexist?', buffer)
        False
        >>> print buffer.len
        0
        >>>

        """

        if self.user or self.passwd:
            authhandler = urllib2.HTTPBasicAuthHandler()
            authhandler.add_password(self.realm, self.urlhost, self.user, self.passwd)
            opener = urllib2.build_opener(authhandler)
            urllib2.install_opener(opener)

        try:
            _url = urllib2.urlopen(url)
            _data = _url.read(100000)
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

def ungzip(outfile, clobber=True, rm=True):
    _outfile = outfile[0:-3]
    if clobber or not os.path.exists(_outfile):
        __outfile = file(_outfile, 'wb')
        gzfile = gzip.open(outfile, 'rb')
        __outfile.write(gzfile.read())
        __outfile.flush()
        __outfile.close()
        if rm:
            os.remove(outfile)
