import os
from os.path import join as pjoin
from urlparse import urlparse
from urllib2 import urlopen
import tempfile

def ensuredirs(path):
    ''' Create directory `path` if it does not already exist
    
    Examples
    --------
    >>> import tempfile, os
    >>> root_path = tempfile.mkdtemp()
    >>> dirname = os.path.join(root_path, 'testdir')
    >>> os.path.exists(dirname)
    False
    >>> ensuredirs(dirname)
    >>> os.path.isdir(dirname)
    True
    >>> os.rmdir(dirname)
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise OSError('"%s" is a file not a directory' % path)


class Cache(object):
    """
    A file cache. The path of the cache can be specified
    or else use ~/.nipy/cache by default.

    The main purpose of the Cache is to cache the contents of URLs.
    """

    def __init__(self, cachepath=None):
        '''
        Examples
        --------
        >>> import tempfile, shutil
        >>> dirname = tempfile.mkdtemp()
        >>> c = Cache(dirname)
        >>> shutil.rmtree(dirname)
        '''
        if cachepath is not None: 
            self.path = cachepath
        elif os.name == 'posix':
            self.path = pjoin(os.environ["HOME"],".nipy","cache")
        elif os.name == 'nt':
            self.path = pjoin(os.environ["HOMEPATH"],".nipy","cache")
        ensuredirs(self.path)

    def tempfile(self,suffix='', prefix=''):
        """ Return an temporary file name in the cache

        Examples
        --------
        >>> import tempfile, shutil, os
        >>> dirname = tempfile.mkdtemp()
        >>> c = Cache(dirname)
        >>> fname = c.tempfile()
        >>> os.path.exists(fname)
        True
        >>> fname = c.tempfile('.nii')
        >>> fname[-4:]
        '.nii'
        >>> shutil.rmtree(dirname)
        """
        _, fname = tempfile.mkstemp(suffix, prefix, self.path)
        return fname

    def filepath(self, uri):
        """
        Return the complete path + filename within the cache.

        Returns
        -------
        filepath : string

        Examples
        --------
        >>> import tempfile, shutil, os
        >>> dirname = tempfile.mkdtemp()
        >>> c = Cache(dirname)
        >>> fname = c.filepath('afile.nii')
        >>> fname == os.path.join(c.path, 'afile.nii')
        True
        >>> shutil.rmtree(dirname)
        """
        (_, netloc, upath, _, _, _) = urlparse(uri)
        if upath.startswith(os.path.sep):
            upath = upath[1:]
        return pjoin(self.path, netloc, upath)

    def filename(self, uri): 
        """
        Return the complete path + filename within the cache.

        :Returns: ``string``
        """
        return self.filepath(uri)
    
    def iscached(self, uri):
        """ Check if a file exists in the cache.

        Parameters
        ----------
        uri : string
           fname relative to the repository
        
        Returns
        -------
        tf : bool
           True if the fname is in the repository
        """
        return os.path.exists(self.filepath(uri))
        
    def cache(self, uri):
        """
        Copy a file into the cache.

        Check if the file is in the cache first.  

        Parameters
        ----------
        uri : filename

        Returns
        -------
        None
        """
        if self.iscached(uri):
            return
        upath = self.filepath(uri)
        pth, fname = os.path.split(upath)
        ensuredirs(pth)
        try:
            openedurl = urlopen(uri)
        except:
            raise IOError("url not found: "+str(uri))
        file(upath, 'w').write(openedurl.read())
        
    def clear(self):
        """ Delete all files in the cache.

        Examples
        --------
        >>> import tempfile, shutil
        >>> dirname = tempfile.mkdtemp()
        >>> c = Cache(dirname)
        >>> fname = c.tempfile()
        >>> os.path.exists(fname)
        True
        >>> c.clear()
        >>> os.path.exists(fname)
        False
        >>> shutil.rmtree(dirname)
        """
        for f in os.listdir(self.path):
            fname = pjoin(self.path, f)
            if os.path.isfile(fname):
                os.unlink(fname)
        
    def retrieve(self, uri):
        """
        Retrieve a file from the cache.
        If not already there, create the file and
        add it to the cache.

        :Returns: ``file``
        """
        self.cache(uri)
        return file(self.filename(uri))
