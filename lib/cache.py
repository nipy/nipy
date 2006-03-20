"""
Create and destroy a per-process cache.
"""

import tempfile, os, atexit, shutil

name = tempfile.mkdtemp(suffix='nipy')

def cleanup():
    """
    Delete the nipy cache.
    """
    shutil.rmtree(name, ignore_errors=True)

def cached():
    """
    Return a tempfile name in the directory neuroimaging.cache.name.
    """
    return tempfile.mkstemp(dir=name)[1]

atexit.register(cleanup)


