"""
Create and destroy a per-process cache.
"""

import tempfile, atexit, shutil

suffix = ''
dirname = tempfile.mkdtemp(suffix=suffix)

def cleanup():
    """
    Delete the nipy cache directory neuroimaging.cache.dirname, registered with atexit.
    """
    shutil.rmtree(dirname, ignore_errors=True)

def cached():
    """
    Return a tempfile name in the directory neuroimaging.cache.dirname.
    """
    return tempfile.mkstemp(dir=dirname)[1]

def cachedir():
    """
    Return a tempfile name in the directory neuroimaging.cache.dirname.
    """
    return tempfile.mkdtemp(dir=dirname)[1]

atexit.register(cleanup)


